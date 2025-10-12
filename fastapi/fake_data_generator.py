"""Utility helpers to create synthetic feature vectors for the gRPC demo."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import os
from datetime import datetime

# Import ETL helpers used during training so that the generated data matches the
# feature space expected by the models registered in MLflow.

# Set ROOT to the project root directory (one level above 'airflow')
ROOT = Path(__file__).resolve().parents[1]
# print(f"Project root directory: {ROOT}")

# Ensure 'airflow/plugins' is in sys.path for etl imports
sys.path.append(os.path.abspath(os.path.join(ROOT, "airflow", "plugins")))
# print(f"sys.path: {sys.path}")

try:
    from etl import (
        cargar_datos,
        eliminar_columnas,
        eliminar_nulos_columna,
        eliminar_nulos_multiples,
        split_dataset,
        imputar_variables,
        codificar_categoricas,
        standard_scaler,
    )
except ModuleNotFoundError as e:
    raise ImportError(
        "Could not import 'etl'. Make sure to run this script from the project root, "
        "so that the 'airflow/plugins' package is accessible."
    ) from e


CATEGORICAL_COLUMNS = ["Gender", "Company Type", "WFH Setup Available"]
COLUMNS_TO_DROP = ["Employee ID", "Date of Joining", "Years in Company"]
TARGET_COLUMN = "Burn Rate"
VARS_TO_IMPUTE = [
    "Designation",
    "Resource Allocation",
    "Mental Fatigue Score",
    "Work Hours per Week",
    "Sleep Hours",
    "Work-Life Balance Score",
    "Manager Support Score",
    "Deadline Pressure Score",
    "Recognition Frequency",
]


def prepare_feature_space(csv_path: Path) -> pd.DataFrame:
    # Add 'mlops2_ceia' folder to sys.path
    mlops2_ceia_path = str(ROOT)
    if mlops2_ceia_path not in sys.path:
        sys.path.append(mlops2_ceia_path)

    dataset = cargar_datos(str(csv_path.parent.resolve()), csv_path.name)
    dataset = eliminar_columnas(dataset, COLUMNS_TO_DROP)
    dataset = eliminar_nulos_columna(dataset, [TARGET_COLUMN])
    dataset = eliminar_nulos_multiples(dataset)

    # The original pipeline splits into train/test; we reuse the same approach so
    # that encoding/scaling parameters mirror what the models saw during
    # training.
    X_train, X_test, _, _ = split_dataset(dataset, 0.2, TARGET_COLUMN, 42)

    _, X_train_imp, X_test_imp = imputar_variables(
        X_train, X_test, VARS_TO_IMPUTE, 10, 42
    )

    _, X_train_enc, X_test_enc = codificar_categoricas(
        X_train_imp, X_test_imp, CATEGORICAL_COLUMNS
    )

    _, X_train_scaled, X_test_scaled = standard_scaler(X_train_enc, X_test_enc)

    combined = pd.concat([X_train_scaled, X_test_scaled], axis=0)
    combined.reset_index(drop=True, inplace=True)
    return combined


class FakeDataGenerator:
    """Samples synthetic feature vectors following the dataset statistics."""

    def __init__(self, feature_space: pd.DataFrame, noise_scale: float = 0.05, seed: Optional[int] = None):
        self.feature_space = feature_space
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.std = feature_space.std().replace(0, 1e-6)

    def sample(self, batch_size: int = 1) -> pd.DataFrame:
        indices = self.rng.integers(0, len(self.feature_space), size=batch_size)
        base = self.feature_space.iloc[indices].reset_index(drop=True)
        noise = self.rng.normal(0.0, self.std.values, size=(batch_size, len(self.std)))
        noisy = base + noise * self.noise_scale
        noisy.columns = self.feature_space.columns
        return noisy

    def stream(self, batch_size: int = 1, limit: Optional[int] = None) -> Iterator[pd.DataFrame]:
        produced = 0
        while limit is None or produced < limit:
            current_batch = batch_size
            if limit is not None:
                current_batch = min(batch_size, limit - produced)
            yield self.sample(current_batch)
            produced += current_batch


def records_from_dataframe(df: pd.DataFrame) -> Iterable[Dict[str, float]]:
    for record in df.to_dict(orient="records"):
        yield {k: float(v) for k, v in record.items()}


def cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate fake feature vectors as JSON lines.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "enriched_employee_dataset.csv",
        help="Path to the source CSV used to fit the feature transformations.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples per emission.")
    parser.add_argument("--limit", type=int, default=None, help="Total number of samples to generate.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1,    # 1 second between batches by default
        help="Optional pause (in seconds) between batches to simulate streaming.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,   # 5% noise by default
        help="Standard deviation multiplier applied to the sampled rows to create variation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args(list(argv) if argv is not None else None)
    feature_space = prepare_feature_space(args.csv)
    print(f"Starting generator with {len(feature_space)} samples in feature space.", file=sys.stderr)
    print(f"- Using CSV at: {args.csv}", file=sys.stderr)
    print(f"- Noise scale: {args.noise_scale}", file=sys.stderr)
    print(f"- Sleep between batches: {args.sleep}", file=sys.stderr)
    print("="*40, file=sys.stderr, end="\n\n")
    generator = FakeDataGenerator(feature_space, noise_scale=args.noise_scale, seed=args.seed)
    try:
        for batch in generator.stream(batch_size=args.batch_size, limit=args.limit):
            for record in records_from_dataframe(batch):
                print(json.dumps(record), end="\n\n")
            if args.sleep:
                time.sleep(args.sleep) # simulate streaming
    except KeyboardInterrupt:
        print(f"\nGenerator interrupted at {datetime.now().isoformat()}.", file=sys.stderr)
    except Exception as e:
        print(f"\nGenerator error at {datetime.now().isoformat()}: {e}", file=sys.stderr)
    else:
        print(f"Generator finished at {datetime.now().isoformat()}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
