"""Inference logic for the fraud-serving API.

Loads the Production model + SHAP explainer once at startup, then for each
request fetches online features, assembles the model's 18 features, scores,
and explains.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from dotenv import load_dotenv
from mlflow import MlflowClient

from src.features.online import get_online_features
from src.serving.schemas import FeatureContribution, PredictionResponse, TransactionRequest
from src.training.train import FEATURE_COLUMNS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "fraud-detector"
MODEL_STAGE = "Production"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
DECISION_THRESHOLD = 0.5
TOP_K = 3

_model: Any = None
_explainer: Any = None
_model_version: str = "unknown"


def load_predictor() -> None:
    """Load the Production model + SHAP explainer once (called at startup)."""
    global _model, _explainer, _model_version
    mlflow.set_tracking_uri(TRACKING_URI)
    _model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    _model_version = MlflowClient().get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0].version
    # tree_path_dependent explainer (no background) — ~50-100x cheaper than
    # interventional and needs no background data, keeping serving latency low.
    _explainer = shap.TreeExplainer(_model)
    logger.info("Loaded model '%s' stage=%s version=%s", MODEL_NAME, MODEL_STAGE, _model_version)


def build_features(req: TransactionRequest, online: dict[str, float]) -> pd.DataFrame:
    """Assemble the model's 18 features from the request + online features."""
    now = datetime.now(UTC)
    row = {
        "amount": req.amount,
        "oldbalanceOrg": req.old_balance_orig,
        "newbalanceOrig": req.new_balance_orig,
        "oldbalanceDest": req.old_balance_dest,
        "newbalanceDest": req.new_balance_dest,
        "errorBalanceOrig": req.new_balance_orig + req.amount - req.old_balance_orig,
        "errorBalanceDest": req.old_balance_dest + req.amount - req.new_balance_dest,
        "balance_delta": req.new_balance_orig - req.old_balance_orig,
        "amount_to_balance_ratio": req.amount / (req.old_balance_orig + 1.0),
        "is_round_amount": int(req.amount % 100 == 0),
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "is_transfer": int(req.transaction_type == "TRANSFER"),
        "dest_tx_count_1h": online.get("dest_tx_count_1h", 0.0),
        "dest_tx_count_24h": online.get("dest_tx_count_24h", 0.0),
        "dest_tx_count_7d": online.get("dest_tx_count_7d", 0.0),
        "dest_amount_sum_24h": online.get("dest_amount_sum_24h", 0.0),
        "dest_amount_avg_24h": online.get("dest_amount_avg_24h", 0.0),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _explain(shap_row: npt.NDArray[np.float64]) -> list[FeatureContribution]:
    """Top-K feature contributions for one prediction."""
    order = np.argsort(np.abs(shap_row))[::-1][:TOP_K]
    return [
        FeatureContribution(
            feature=FEATURE_COLUMNS[int(i)],
            shap_value=round(float(shap_row[int(i)]), 4),
            direction="increases_fraud_risk" if shap_row[int(i)] > 0 else "decreases_fraud_risk",
        )
        for i in order
    ]


def predict(req: TransactionRequest) -> PredictionResponse:
    """Score one transaction with a SHAP explanation."""
    if _model is None or _explainer is None:
        load_predictor()
    start = time.perf_counter()
    features = build_features(req, get_online_features(req.destination_id))
    proba = float(np.asarray(_model.predict_proba(features))[0, 1])
    shap_row = np.asarray(_explainer.shap_values(features), dtype=np.float64)[0]
    decision = "FRAUD" if proba >= DECISION_THRESHOLD else "LEGITIMATE"
    return PredictionResponse(
        fraud_probability=round(proba, 6),
        decision=decision,
        decision_threshold=DECISION_THRESHOLD,
        shap_explanation=_explain(shap_row),
        model_version=_model_version,
        latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
    )
