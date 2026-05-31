"""Drift-triggered retraining.

Usage:
    python -m src.monitoring.retrain_trigger --report reports/drift_latest.json
    # decide only, no actual retrain (testing):
    python -m src.monitoring.retrain_trigger --report ... --no-train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CMD = ["python", "-m", "src.training.train"]
DEFAULT_THRESHOLD = 0.20
EXPERIMENT = "drift-monitoring"


def evaluate_drift(summary: dict[str, Any], threshold: float) -> tuple[bool, float, int]:
    """Return (triggered, share, count) from the DriftedColumnsCount metric."""
    for m in summary.get("metrics", []):
        if str(m.get("metric_name", "")).startswith("DriftedColumnsCount"):
            v = m["value"]
            share = float(v["share"])
            return share > threshold, share, int(v["count"])
    return False, 0.0, 0


def run_training(cmd: list[str]) -> int:
    """Shell out to the configured training command."""
    logger.info("Triggering retraining: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _tracking_server_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Quick TCP probe so we skip MLflow fast when the server is down."""
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def log_to_mlflow(
    share: float, count: int, threshold: float, triggered: bool, ret: int | None
) -> None:
    """Log the drift-trigger event to MLflow (best-effort, fast-fail if down)."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    if not _tracking_server_reachable(tracking_uri):
        raise ConnectionError(f"MLflow tracking server not reachable at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="drift-trigger"):
        mlflow.set_tag("trigger", "drift")
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("triggered_retraining", triggered)
        if ret is not None:
            mlflow.log_param("retrain_returncode", ret)
        mlflow.log_metric("drifted_share", share)
        mlflow.log_metric("drifted_count", count)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fire retraining when drift exceeds threshold.")
    parser.add_argument("--report", required=True, type=Path, help="Drift JSON from drift.py")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--train-cmd", nargs="+", default=DEFAULT_TRAIN_CMD)
    parser.add_argument("--no-train", action="store_true", help="Decide only; skip training.")
    args = parser.parse_args()

    summary = json.loads(args.report.read_text())
    triggered, share, count = evaluate_drift(summary, args.threshold)
    logger.info(
        "Drift: share=%.2f count=%d, threshold=%.2f → %s",
        share,
        count,
        args.threshold,
        "TRIGGER" if triggered else "no action",
    )

    ret: int | None = None
    if triggered and not args.no_train:
        ret = run_training(args.train_cmd)
        logger.info("retrain exit code: %d", ret)

    try:
        log_to_mlflow(share, count, args.threshold, triggered, ret)
        logger.info("Logged drift-trigger event to MLflow.")
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)

    return 0 if not triggered else (ret or 0)


if __name__ == "__main__":
    sys.exit(main())
