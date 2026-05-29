"""Train the baseline fraud-detection model.

Usage:
    python -m src.training.train

Loads engineered features, does a time-based split, trains XGBoost, evaluates
(AUC-PR primary), logs to MLflow, and — if it clears the promotion gate —
registers the model as 'fraud-detector'.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature

from src.training.evaluate import (
    compute_metrics,
    passes_promotion_gate,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_shap_summary,
)
from src.training.model import FraudModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training.yaml")
DECISION_THRESHOLD = 0.5  # baseline cutoff; tuned properly in Week 9

FEATURE_COLUMNS: list[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "balance_delta",
    "amount_to_balance_ratio",
    "is_round_amount",
    "hour_of_day",
    "day_of_week",
    "dest_tx_count_1h",
    "dest_tx_count_24h",
    "dest_tx_count_7d",
    "dest_amount_sum_24h",
    "dest_amount_avg_24h",
    "is_transfer",
]
LABEL = "isFraud"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the training config YAML."""
    with path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def load_features(path: Path) -> pd.DataFrame:
    """Load the engineered Parquet and add the encoded transaction type."""
    df = pd.read_parquet(path)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: earliest steps -> train, latest -> test. Never random."""
    cutoff = df["step"].quantile(1.0 - test_fraction)
    return df[df["step"] <= cutoff], df[df["step"] > cutoff]


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into feature matrix X and label vector y."""
    return df[FEATURE_COLUMNS], df[LABEL]


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Negatives-to-positives ratio for XGBoost imbalance weighting."""
    n_pos = int(y.sum())
    return (len(y) - n_pos) / max(n_pos, 1)


def main() -> int:
    """Full training pipeline entry point."""
    load_dotenv()
    cfg = load_config()

    df = load_features(Path(cfg["paths"]["processed_data"]))
    train_df, test_df = time_based_split(df, float(cfg["split"]["test_fraction"]))
    X_train, y_train = to_xy(train_df)
    X_test, y_test = to_xy(test_df)

    scale_pos_weight = compute_scale_pos_weight(y_train)
    logger.info("scale_pos_weight = %.1f", scale_pos_weight)

    model = FraudModel(params=cfg["model"], scale_pos_weight=scale_pos_weight)
    logger.info("Training on %s rows ...", f"{len(X_train):,}")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)
    y_true = np.asarray(y_test.to_numpy(), dtype=np.int_)
    y_pred = np.asarray(y_proba >= DECISION_THRESHOLD, dtype=np.int_)
    metrics = compute_metrics(y_true, y_proba, float(cfg["thresholds"]["recall_target"]))
    logger.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    mlflow.set_tracking_uri(str(cfg["paths"]["mlflow_tracking_uri"]))
    mlflow.set_experiment(str(cfg["paths"]["experiment_name"]))

    with mlflow.start_run():
        mlflow.log_params(cfg["model"])
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = Path(tmp) / "confusion_matrix.png"
            fi_path = Path(tmp) / "feature_importance.png"
            plot_confusion_matrix(y_true, y_pred, cm_path)
            plot_feature_importance(model.feature_importances(), fi_path)
            shap_path = Path(tmp) / "shap_summary.png"
            X_shap = X_test.sample(min(2000, len(X_test)), random_state=42)
            plot_shap_summary(model.shap_values(X_shap), X_shap, shap_path)
            mlflow.log_artifact(str(shap_path))
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(fi_path))

        min_auc_pr = float(cfg["thresholds"]["promotion_auc_pr"])
        if not passes_promotion_gate(metrics, min_auc_pr):
            logger.error(
                "❌ Promotion gate FAILED: auc_pr=%.4f < %.2f — not registering.",
                metrics["auc_pr"],
                min_auc_pr,
            )
            return 1

        signature = infer_signature(X_test, y_proba)
        mlflow.xgboost.log_model(
            model.model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(3),
            registered_model_name=str(cfg["paths"]["registered_model_name"]),
        )
        logger.info(
            "✅ Registered '%s' (auc_pr=%.4f).",
            cfg["paths"]["registered_model_name"],
            metrics["auc_pr"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
