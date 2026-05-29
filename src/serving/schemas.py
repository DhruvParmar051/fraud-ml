"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel


class TransactionRequest(BaseModel):
    """An incoming transaction to score."""

    customer_id: str  # originating account (nameOrig)
    destination_id: str  # destination/merchant account (nameDest) — Feast lookup key
    transaction_type: str  # "TRANSFER" or "CASH_OUT"
    amount: float
    old_balance_orig: float
    new_balance_orig: float
    old_balance_dest: float
    new_balance_dest: float


class FeatureContribution(BaseModel):
    """One feature's SHAP contribution to a prediction."""

    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    """The scoring result returned to the caller."""

    fraud_probability: float
    decision: str  # "FRAUD" | "LEGITIMATE"
    decision_threshold: float
    shap_explanation: list[FeatureContribution]
    model_version: str
    latency_ms: float
