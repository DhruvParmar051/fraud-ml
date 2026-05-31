"""Validate raw PaySim data with Great Expectations (GE 1.x API).

Usage:
    python -m src.data.validate --input data/raw/paysim.csv

Exit codes:
    0 = all expectations passed
    1 = validation failed, or the input file could not be read

Each run is also logged (best-effort) to the MLflow 'data-validation' experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import great_expectations as gx
import mlflow
import pandas as pd
from dotenv import load_dotenv
from great_expectations import expectations as gxe
from great_expectations.core import ExpectationSuiteValidationResult

load_dotenv()  # read .env so MLflow + MinIO connection settings are available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Real PaySim schema. NOTE: last column is `isFlaggedFraud`, not `isFraudster`
# as the project brief stated — confirmed during EDA.
EXPECTED_COLUMNS: list[str] = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]
VALID_TYPES: list[str] = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
SUITE_NAME = "paysim_raw"
MLFLOW_EXPERIMENT = "data-validation"


def build_expectation_suite() -> gx.ExpectationSuite:
    """Define the expectation suite encoding our EDA invariants."""
    suite = gx.ExpectationSuite(name=SUITE_NAME)

    # Schema: every expected column must be present.
    for column in EXPECTED_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnToExist(column=column))

    # amount: present and non-negative (PaySim contains some amount == 0 rows).
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="amount"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="amount", min_value=0, strict_min=False)
    )

    # isFraud: present and binary {0, 1}.
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="isFraud"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="isFraud", value_set=[0, 1]))

    # The "type" column: one of the five known transaction types.
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="type", value_set=VALID_TYPES))

    # Natural key uniqueness: no duplicate transactions.
    suite.add_expectation(
        gxe.ExpectCompoundColumnsToBeUnique(column_list=["step", "nameOrig", "nameDest", "amount"])
    )
    return suite


def validate_dataframe(df: pd.DataFrame) -> ExpectationSuiteValidationResult:
    """Run the suite against ``df`` and return the full GE result object."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="paysim_pandas")
    asset = data_source.add_dataframe_asset(name="paysim_raw_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = build_expectation_suite()
    return batch.validate(suite)


def _tracking_server_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Quick TCP check so we skip MLflow fast when the server is down."""
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def log_to_mlflow(results: ExpectationSuiteValidationResult) -> None:
    """Log validation metrics and the full result JSON to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    if not _tracking_server_reachable(tracking_uri):
        raise ConnectionError(f"MLflow tracking server not reachable at {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    n_total = len(results.results)
    n_passed = sum(1 for r in results.results if r.success)

    with mlflow.start_run(run_name=SUITE_NAME):
        mlflow.log_param("suite", SUITE_NAME)
        mlflow.log_param("n_expectations", n_total)
        mlflow.log_metric("expectations_passed", n_passed)
        mlflow.log_metric("success", int(bool(results.success)))

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "validation_result.json"
            artifact.write_text(json.dumps(results.to_json_dict(), indent=2, default=str))
            mlflow.log_artifact(str(artifact))


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate raw PaySim data.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the raw PaySim CSV.")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    logger.info("Loading %s ...", args.input)
    df = pd.read_csv(args.input)
    logger.info("Loaded %s rows x %d columns", f"{len(df):,}", df.shape[1])

    logger.info("Running expectation suite '%s' ...", SUITE_NAME)
    results = validate_dataframe(df)

    n_total = len(results.results)
    n_passed = sum(1 for r in results.results if r.success)
    logger.info("Expectations passed: %d/%d", n_passed, n_total)
    if not results.success:
        for r in results.results:
            if not r.success and r.expectation_config is not None:
                logger.error("  FAILED: %s", r.expectation_config.type)

    # MLflow logging must never break validation — best-effort only.
    try:
        log_to_mlflow(results)
        logger.info("Logged run to MLflow experiment '%s'.", MLFLOW_EXPERIMENT)
    except Exception as exc:
        logger.warning("Could not log to MLflow (continuing): %s", exc)

    if results.success:
        logger.info("✅ Validation PASSED — all expectations met.")
        return 0
    logger.error("❌ Validation FAILED — see failures above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
