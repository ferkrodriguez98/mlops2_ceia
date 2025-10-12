"""gRPC server that serves MLflow models for real-time inference."""

from __future__ import annotations

import logging
import os
from concurrent import futures
from typing import Any, Dict, Iterable, Optional, Tuple

import grpc
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from grpc_service import prediction_pb2, prediction_pb2_grpc

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MODEL_EXPERIMENT", "modelos_optimizados")
DEFAULT_MODEL_ALIAS = os.getenv("DEFAULT_MODEL_ALIAS", "knn")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
CLIENT = MlflowClient()

MODEL_NAME_MAP = {
    "knn": "Knn_Classifier",
    "svm": "SVC",
    "lightgbm": "LightGBM_Classifier",
}

MODEL_CACHE: Dict[str, Tuple[Any, Dict[str, Any], Dict[str, Any]]] = {}


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded from the MLflow registry."""


def _resolve_registered_name(alias: str) -> str:
    name = MODEL_NAME_MAP.get(alias.lower())
    if not name:
        raise ModelLoadError(f"Unknown model alias '{alias}'.")
    return name


def _get_latest_version_entry(reg_name: str):
    versions = CLIENT.get_latest_versions(reg_name)
    if not versions:
        all_versions = CLIENT.search_model_versions(f"name='{reg_name}'")
        if not all_versions:
            raise ModelLoadError(f"No versions found for registered model '{reg_name}'.")
        versions = list(all_versions)

    try:
        best = max(versions, key=lambda v: int(v.version))
    except Exception:  # pragma: no cover - safety fallback
        best = max(versions, key=lambda v: str(v.version))
    return best


def _load_model_from_registry(alias: str):
    reg_name = _resolve_registered_name(alias)

    try:
        CLIENT.get_registered_model(reg_name)
    except Exception as exc:  # pragma: no cover - passthrough
        raise ModelLoadError(
            f"Registered model '{reg_name}' not found in MLflow Registry."
        ) from exc

    latest = _get_latest_version_entry(reg_name)
    version = str(latest.version)
    cache_key = f"{alias}::{version}"

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    model_uri = f"models:/{reg_name}/{version}"
    LOGGER.info("Loading model for alias=%s from URI=%s", alias, model_uri)

    try:
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:  # pragma: no cover - passthrough
        raise ModelLoadError(f"Failed to load '{model_uri}': {exc}") from exc

    registry_info = {
        "registered_name": reg_name,
        "version": version,
        "current_stage": getattr(latest, "current_stage", None),
        "run_id": latest.run_id,
        "status": getattr(latest, "status", None),
        "source": getattr(latest, "source", None),
    }

    run_info = {"metrics": {}, "params": {}}
    try:
        run = CLIENT.get_run(latest.run_id)
        run_info["metrics"] = run.data.metrics or {}
        run_info["params"] = run.data.params or {}
    except Exception:  # pragma: no cover - optional metadata
        pass

    MODEL_CACHE[cache_key] = (pyfunc_model, registry_info, run_info)
    return MODEL_CACHE[cache_key]


def _features_to_dataframe(instances: Iterable[prediction_pb2.FeatureVector]) -> pd.DataFrame:
    rows = []
    for item in instances:
        rows.append(dict(item.values))
    if not rows:
        raise ValueError("No feature vectors provided.")
    return pd.DataFrame(rows)


class PredictionService(prediction_pb2_grpc.PredictionServiceServicer):
    """Concrete implementation of the PredictionService gRPC API."""

    def __init__(self, default_model: str = DEFAULT_MODEL_ALIAS):
        self.default_model = default_model

    def _ensure_model(self, alias: Optional[str]):
        selected_alias = alias or self.default_model
        model, registry_info, _run = _load_model_from_registry(selected_alias)
        return selected_alias, model, registry_info

    def Predict(self, request, context):  # noqa: N802 (gRPC signature)
        try:
            alias, model, registry_info = self._ensure_model(request.model_alias)
        except ModelLoadError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        try:
            df = _features_to_dataframe(request.instances)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - unexpected errors
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        try:
            preds = model.predict(df)
        except Exception as exc:  # pragma: no cover - model errors
            context.abort(grpc.StatusCode.INTERNAL, f"Model inference failed: {exc}")

        return prediction_pb2.PredictResponse(
            alias=alias,
            registered_name=registry_info["registered_name"],
            version=registry_info["version"],
            predictions=[
                prediction_pb2.Prediction(predicted_class=int(value)) for value in preds
            ],
        )

    def PredictStream(self, request_iterator, context):  # noqa: N802
        alias: Optional[str] = None
        model = None
        registry_info: Optional[Dict[str, Any]] = None

        for request in request_iterator:
            try:
                if request.model_alias and request.model_alias != alias:
                    alias, model, registry_info = self._ensure_model(request.model_alias)
                elif model is None:
                    alias, model, registry_info = self._ensure_model(request.model_alias or alias)
            except ModelLoadError as exc:
                context.abort(grpc.StatusCode.NOT_FOUND, str(exc))

            try:
                df = _features_to_dataframe([request.instance])
            except ValueError as exc:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception as exc:  # pragma: no cover - unexpected errors
                context.abort(grpc.StatusCode.INTERNAL, str(exc))

            try:
                pred = model.predict(df)[0]
            except Exception as exc:  # pragma: no cover
                context.abort(grpc.StatusCode.INTERNAL, f"Model inference failed: {exc}")

            yield prediction_pb2.StreamPredictResponse(
                alias=alias,
                registered_name=registry_info["registered_name"],
                version=registry_info["version"],
                prediction=prediction_pb2.Prediction(predicted_class=int(pred)),
            )


def serve(host: str = "0.0.0.0", port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionService(), server)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    LOGGER.info("Starting gRPC server on %s", listen_addr)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
