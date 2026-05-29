"""XGBoost model wrapper for fraud detection, with SHAP explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
import xgboost as xgb

N_BACKGROUND = 100  # SHAP background sample size — small, for speed


class FraudModel:
    """XGBoost classifier wrapper with SHAP-based explanations."""

    def __init__(self, params: dict[str, Any], scale_pos_weight: float) -> None:
        self.feature_names: list[str] = []
        self.model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        self._explainer: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model, remember feature order, and build the SHAP explainer."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        background = X.sample(min(N_BACKGROUND, len(X)), random_state=42)
        self._explainer = shap.TreeExplainer(
            self.model, data=background, feature_perturbation="interventional"
        )

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return P(fraud) for each row (the positive-class probability)."""
        proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        return proba[:, 1]

    def feature_importances(self) -> dict[str, float]:
        """Return XGBoost feature importances, highest first."""
        importances = np.asarray(self.model.feature_importances_, dtype=np.float64)
        pairs = zip(self.feature_names, importances, strict=True)
        return {n: float(v) for n, v in sorted(pairs, key=lambda kv: kv[1], reverse=True)}

    def shap_values(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """SHAP values for X, shape (n_rows, n_features)."""
        if self._explainer is None:
            raise RuntimeError("Model must be fit before computing SHAP values.")
        return np.asarray(self._explainer.shap_values(X), dtype=np.float64)

    def explain(self, X: pd.DataFrame, top_k: int = 3) -> dict[str, list[dict[str, Any]]]:
        """Top-k feature contributions for the FIRST row of X.

        Returns ``{"top_features": [{"feature", "shap_value", "direction"}, ...]}``.
        """
        row = self.shap_values(X)[0]
        order = np.argsort(np.abs(row))[::-1][:top_k]
        top_features = [
            {
                "feature": self.feature_names[int(i)],
                "shap_value": round(float(row[int(i)]), 4),
                "direction": (
                    "increases_fraud_risk" if row[int(i)] > 0 else "decreases_fraud_risk"
                ),
            }
            for i in order
        ]
        return {"top_features": top_features}
