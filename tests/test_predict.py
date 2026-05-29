"""Unit tests for prediction explanation logic (no infra required)."""

from __future__ import annotations

import numpy as np

from src.serving.predict import _explain
from src.training.train import FEATURE_COLUMNS


def test_explain_returns_top3_sorted_by_magnitude() -> None:
    shap_row = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    shap_row[0] = 5.0
    shap_row[1] = -3.0
    shap_row[2] = 1.0
    out = _explain(shap_row)
    assert len(out) == 3
    assert out[0].feature == FEATURE_COLUMNS[0]
    assert out[0].direction == "increases_fraud_risk"
    assert out[1].direction == "decreases_fraud_risk"
    assert abs(out[0].shap_value) >= abs(out[1].shap_value) >= abs(out[2].shap_value)
