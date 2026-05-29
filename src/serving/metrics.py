"""Prometheus metrics for the serving API."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

FRAUD_PREDICTIONS = Counter("fraud_predictions_total", "Total predictions made", ["decision"])
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25),
)
MODEL_VERSION_INFO = Gauge("model_version_info", "Currently served model version", ["version"])
