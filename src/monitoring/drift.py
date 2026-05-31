"""Evidently drift report: reference vs current features.

Usage:
    python -m src.monitoring.drift \
        --reference data/reference/training_features.parquet \
        --current data/predictions/recent.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, data_definition=DataDefinition())


def generate_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path
) -> dict[str, Any]:
    """Run drift + data-summary presets; save HTML + JSON; return the summary dict."""
    snap = Report([DataDriftPreset(), DataSummaryPreset()]).run(
        current_data=_dataset(current), reference_data=_dataset(reference)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    snap.save_html(str(out_dir / f"drift_{stamp}.html"))
    summary: dict[str, Any] = snap.dict()
    (out_dir / f"drift_{stamp}.json").write_text(json.dumps(summary, default=str, indent=2))
    logger.info("Wrote drift_%s.html and .json to %s", stamp, out_dir)
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate an Evidently drift report.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    summary = generate_drift_report(
        pd.read_parquet(args.reference), pd.read_parquet(args.current), args.output
    )
    for metric in summary["metrics"]:
        if str(metric["metric_name"]).startswith("DriftedColumnsCount"):
            value = metric["value"]
            logger.info("Drifted columns: %s (share %.2f)", value["count"], value["share"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
