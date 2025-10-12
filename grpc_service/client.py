"""Simple CLI client that consumes the gRPC prediction service."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import grpc

from grpc_service import prediction_pb2, prediction_pb2_grpc
from grpc_service.fake_data_generator import (
    FakeDataGenerator,
    prepare_feature_space,
    records_from_dataframe,
)


def _build_feature_vectors(batch) -> Iterable[prediction_pb2.FeatureVector]:
    for record in records_from_dataframe(batch):
        yield prediction_pb2.FeatureVector(values=record)


def _stream_requests(
    generator: FakeDataGenerator,
    model_alias: str,
    batch_size: int,
    limit: Optional[int],
    interval: float,
) -> Iterator[prediction_pb2.StreamPredictRequest]:
    produced = 0
    for batch in generator.stream(batch_size=batch_size, limit=limit):
        for vector in _build_feature_vectors(batch):
            yield prediction_pb2.StreamPredictRequest(
                model_alias=model_alias,
                instance=vector,
            )
            produced += 1
            if limit and produced >= limit:
                return
            if interval:
                time.sleep(interval)


def cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interact with the prediction gRPC service.")
    parser.add_argument("--host", default="localhost", help="gRPC server host.")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port.")
    parser.add_argument("--model", default="knn", help="Model alias to use (knn, svm, lightgbm).")
    parser.add_argument(
        "--mode",
        choices=["batch", "stream"],
        default="batch",
        help="Call Predict with batches or use the bidirectional streaming RPC.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/enriched_employee_dataset.csv"),
        help="Dataset used to estimate the feature distribution.",
    )
    parser.add_argument("--batch-size", type=int, default=5, help="Samples per Predict call.")
    parser.add_argument("--limit", type=int, default=None, help="Total samples to send.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Delay between streaming messages (seconds).",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,
        help="Noise multiplier passed to the fake data generator.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args(list(argv) if argv is not None else None)

    feature_space = prepare_feature_space(args.csv)
    generator = FakeDataGenerator(feature_space, noise_scale=args.noise_scale, seed=args.seed)

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = prediction_pb2_grpc.PredictionServiceStub(channel)

    if args.mode == "batch":
        produced = 0
        for batch in generator.stream(batch_size=args.batch_size, limit=args.limit):
            request = prediction_pb2.PredictRequest(
                model_alias=args.model,
                instances=list(_build_feature_vectors(batch)),
            )
            response = stub.Predict(request)
            payload = {
                "alias": response.alias,
                "version": response.version,
                "predictions": [pred.predicted_class for pred in response.predictions],
            }
            print(json.dumps(payload))
            produced += len(payload["predictions"])
            if args.limit is not None and produced >= args.limit:
                break
    else:
        responses = stub.PredictStream(
            _stream_requests(
                generator,
                model_alias=args.model,
                batch_size=1,
                limit=args.limit,
                interval=args.interval,
            )
        )
        for response in responses:
            payload = {
                "alias": response.alias,
                "version": response.version,
                "prediction": response.prediction.predicted_class,
            }
            print(json.dumps(payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
