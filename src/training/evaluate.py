"""Evaluation metrics, plots, and the promotion gate for the fraud model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — render to file, no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


def precision_at_recall(precision: FloatArray, recall: FloatArray, target_recall: float) -> float:
    """Best precision achievable while keeping recall >= target_recall."""
    mask = recall >= target_recall
    if not mask.any():
        return 0.0
    return float(precision[mask].max())


def compute_metrics(
    y_true: IntArray, y_proba: FloatArray, recall_target: float
) -> dict[str, float]:
    """Compute fraud-relevant metrics. AUC-PR is the primary metric."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "precision_at_target_recall": precision_at_recall(
            np.asarray(precision, dtype=np.float64),
            np.asarray(recall, dtype=np.float64),
            recall_target,
        ),
    }


def passes_promotion_gate(metrics: dict[str, float], min_auc_pr: float) -> bool:
    """The model must clear the AUC-PR bar to be registerable."""
    return metrics["auc_pr"] >= min_auc_pr


def plot_confusion_matrix(y_true: IntArray, y_pred: IntArray, out_path: Path) -> None:
    """Save a confusion-matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(importances: dict[str, float], out_path: Path) -> None:
    """Save a horizontal bar chart of feature importances (highest at top)."""
    names = list(importances.keys())[::-1]
    values = list(importances.values())[::-1]
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(names, values)
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
