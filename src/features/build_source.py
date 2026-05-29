"""Build the Feast offline feature source from the engineered Parquet.

Adds a synthetic event_timestamp (derived from `step`) and keeps the
destination-keyed window features, writing a Feast-ready Parquet.

Usage:
    python -m src.features.build_source
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DATE = pd.Timestamp("2024-01-01", tz="UTC")
SOURCE_PARQUET = Path("data/processed/features.parquet")
FEAST_SOURCE = Path("src/features/feature_repo/data/destination_stats.parquet")
DEST_FEATURES = [
    "dest_tx_count_1h",
    "dest_tx_count_24h",
    "dest_tx_count_7d",
    "dest_amount_sum_24h",
    "dest_amount_avg_24h",
]


def main() -> int:
    """Build the Feast source Parquet with an event_timestamp."""
    logger.info("Reading %s ...", SOURCE_PARQUET)
    df = pd.read_parquet(SOURCE_PARQUET)

    # `step` is hours since the simulation start — turn it into a real timestamp.
    df["event_timestamp"] = BASE_DATE + pd.to_timedelta(df["step"], unit="h")

    out = df[["nameDest", "event_timestamp", *DEST_FEATURES]].copy()
    FEAST_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEAST_SOURCE, index=False)
    logger.info(
        "Wrote %s rows to %s (%s .. %s)",
        f"{len(out):,}",
        FEAST_SOURCE,
        out["event_timestamp"].min(),
        out["event_timestamp"].max(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
