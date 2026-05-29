"""XGBoost model wrapper for fraud detection.

(SHAP explanations are added in Week 7.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb


class FraudModel:
    """Thin wrapper around an XGBoost classifier for fraud scoring."""

    def __init__(self, params: dict[str, Any], scale_pos_weight: float) -> None:
        self.feature_names: list[str] = []
        self.model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model and remember the feature order for serving."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return P(fraud) for each row (the positive-class probability)."""
        proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        return proba[:, 1]

    def feature_importances(self) -> dict[str, float]:
        """Return feature importances, highest first."""
        importances = np.asarray(self.model.feature_importances_, dtype=np.float64)
        pairs = zip(self.feature_names, importances, strict=True)
        return {n: float(v) for n, v in sorted(pairs, key=lambda kv: kv[1], reverse=True)}
