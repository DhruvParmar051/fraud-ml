"""Compare class-imbalance strategies for the fraud model (Week 9).

Usage:
    python -m src.training.imbalance

Compares three approaches under one MLflow experiment:
  1. scale_pos_weight  — XGBoost native class weighting
  2. SMOTE             — synthetic minority oversampling
  3. threshold tuning  — operating-point selection (does not change ranking)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.training.evaluate import compute_metrics
from src.training.model import FraudModel
from src.training.train import (
    compute_scale_pos_weight,
    load_config,
    load_features,
    time_based_split,
    to_xy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EXPERIMENT = "imbalance-comparison"
SMOTE_RATIO = 0.1  # minority:majority ratio after oversampling

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]


def threshold_for_recall(y_true: IntArray, y_proba: FloatArray, target: float) -> float:
    """Highest decision threshold that still achieves >= target recall."""
    _, recall, thresholds = precision_recall_curve(y_true, y_proba)
    mask = recall[:-1] >= target
    if not mask.any():
        return 0.5
    return float(np.asarray(thresholds)[mask].max())


def metrics_at_threshold(y_true: IntArray, y_proba: FloatArray, thr: float) -> dict[str, float]:
    """Precision/recall/F1 at a specific decision threshold."""
    y_pred = (y_proba >= thr).astype(int)
    return {
        "threshold": thr,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> int:
    """Run the three-way imbalance comparison."""
    load_dotenv()
    cfg = load_config()

    df = load_features(Path(cfg["paths"]["processed_data"]))
    train_df, test_df = time_based_split(df, float(cfg["split"]["test_fraction"]))
    tr_df, val_df = time_based_split(train_df, 0.20)
    X_tr, y_tr = to_xy(tr_df)
    X_val, y_val = to_xy(val_df)
    X_te, y_te = to_xy(test_df)
    y_te_arr = np.asarray(y_te.to_numpy(), dtype=np.int_)
    y_val_arr = np.asarray(y_val.to_numpy(), dtype=np.int_)
    recall_target = float(cfg["thresholds"]["recall_target"])
    params = cfg["model"]

    mlflow.set_tracking_uri(str(cfg["paths"]["mlflow_tracking_uri"]))
    mlflow.set_experiment(EXPERIMENT)

    # 1. scale_pos_weight
    spw = compute_scale_pos_weight(y_tr)
    spw_model = FraudModel(params=params, scale_pos_weight=spw)
    spw_model.fit(X_tr, y_tr)
    spw_proba = spw_model.predict_proba(X_te)
    spw_metrics = compute_metrics(y_te_arr, spw_proba, recall_target)
    with mlflow.start_run(run_name="scale_pos_weight"):
        mlflow.log_param("strategy", "scale_pos_weight")
        mlflow.log_metrics(spw_metrics)

    # 2. SMOTE
    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=42)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    X_res = pd.DataFrame(X_res, columns=X_tr.columns)
    smote_model = FraudModel(params=params, scale_pos_weight=1.0)
    smote_model.fit(X_res, pd.Series(y_res))
    smote_metrics = compute_metrics(y_te_arr, smote_model.predict_proba(X_te), recall_target)
    with mlflow.start_run(run_name="smote"):
        mlflow.log_param("strategy", "smote")
        mlflow.log_param("smote_ratio", SMOTE_RATIO)
        mlflow.log_metrics(smote_metrics)

    # 3. threshold tuning on the scale_pos_weight model (tuned on val, applied to test)
    thr = threshold_for_recall(y_val_arr, spw_model.predict_proba(X_val), recall_target)
    thr_metrics = metrics_at_threshold(y_te_arr, spw_proba, thr)
    with mlflow.start_run(run_name="threshold_tuned"):
        mlflow.log_param("strategy", "threshold_tuned")
        mlflow.log_metrics(thr_metrics)

    logger.info("scale_pos_weight: %s", {k: round(v, 4) for k, v in spw_metrics.items()})
    logger.info("smote:            %s", {k: round(v, 4) for k, v in smote_metrics.items()})
    logger.info("threshold_tuned:  %s", {k: round(v, 4) for k, v in thr_metrics.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
