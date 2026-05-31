"""PySpark ETL: raw PaySim CSV -> engineered feature Parquet.

Usage:
    python -m src.data.etl --input data/raw/paysim.csv --output data/processed/

Filters to TRANSFER and CASH_OUT (the only fraud-bearing types), engineers
row-level and destination-window features, and writes Parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]
HOURS_1H = 1
HOURS_24H = 24
HOURS_7D = 24 * 7


def create_spark_session(app_name: str = "paysim-etl") -> SparkSession:
    """Create a local-mode Spark session tuned for a single laptop."""
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def add_row_features(df: DataFrame) -> DataFrame:
    """Add features computable from a single row — no history, no leakage."""
    return (
        df.withColumn(
            "errorBalanceOrig",
            F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"),
        )
        .withColumn(
            "errorBalanceDest",
            F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"),
        )
        .withColumn("balance_delta", F.col("newbalanceOrig") - F.col("oldbalanceOrg"))
        .withColumn(
            "amount_to_balance_ratio",
            F.col("amount") / (F.col("oldbalanceOrg") + F.lit(1.0)),
        )
        .withColumn("is_round_amount", (F.col("amount") % 100 == 0).cast("int"))
        .withColumn("hour_of_day", (F.col("step") % 24).cast("int"))
        .withColumn("day_of_week", (F.floor(F.col("step") / 24) % 7).cast("int"))
    )


def add_window_features(df: DataFrame) -> DataFrame:
    """Add destination-keyed, time-windowed aggregates.

    Windows are ordered by ``step`` and span only PAST hours (negative range to
    0), so a row never sees future transactions — point-in-time safe. Keyed on
    ``nameDest`` because destination (mule) accounts repeat, unlike one-shot
    originators (an EDA finding).
    """
    w_1h = Window.partitionBy("nameDest").orderBy("step").rangeBetween(-HOURS_1H, 0)
    w_24h = Window.partitionBy("nameDest").orderBy("step").rangeBetween(-HOURS_24H, 0)
    w_7d = Window.partitionBy("nameDest").orderBy("step").rangeBetween(-HOURS_7D, 0)
    return (
        df.withColumn("dest_tx_count_1h", F.count(F.lit(1)).over(w_1h))
        .withColumn("dest_tx_count_24h", F.count(F.lit(1)).over(w_24h))
        .withColumn("dest_tx_count_7d", F.count(F.lit(1)).over(w_7d))
        .withColumn("dest_amount_sum_24h", F.sum("amount").over(w_24h))
        .withColumn("dest_amount_avg_24h", F.avg("amount").over(w_24h))
    )


def run_etl(spark: SparkSession, input_path: Path, output_path: Path) -> int:
    """Run the full ETL, write Parquet to ``<output>/features.parquet``.

    Returns the number of rows written.
    """
    logger.info("Reading %s ...", input_path)
    df = spark.read.csv(str(input_path), header=True, inferSchema=True)

    df = df.filter(F.col("type").isin(FRAUD_TYPES))
    df = add_row_features(df)
    df = add_window_features(df)

    parquet_path = output_path / "features.parquet"
    logger.info("Writing Parquet to %s ...", parquet_path)
    df.write.mode("overwrite").parquet(str(parquet_path))

    n_rows = int(df.count())
    logger.info("✅ Wrote %s rows x %d columns.", f"{n_rows:,}", len(df.columns))
    return n_rows


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="PySpark ETL for PaySim.")
    parser.add_argument("--input", required=True, type=Path, help="Raw PaySim CSV path.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    spark = create_spark_session()
    try:
        run_etl(spark, args.input, args.output)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
