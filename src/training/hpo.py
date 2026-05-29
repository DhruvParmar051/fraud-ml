"""Optuna hyperparameter optimization for the fraud model.

Usage:
    python -m src.training.hpo

Runs a TPE study (30 trials) on a time-based validation split, logging every
trial as a nested MLflow run under a parent 'hpo' run. Retrains the best params
on the full training set, evaluates on the held-out test set, and registers the
tuned model if it clears the promotion gate.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
from dotenv import load_dotenv
from mlflow.models import infer_signature

from src.training.evaluate import compute_metrics, passes_promotion_gate
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

N_TRIALS = 30
VALIDATION_FRACTION = 0.20  # carved from the training period, time-based


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one hyperparameter configuration from the search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "eval_metric": "aucpr",
        "random_state": 42,
    }


def log_trial_to_wandb(params: dict[str, Any], val_auc_pr: float) -> None:
    """Log one HPO trial as a W&B run, for parallel-coordinates / sweep viz."""
    import wandb

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "fraud-detection"),
        group="hpo",
        config=params,
        reinit="finish_previous",
    )
    wandb.log({"val_auc_pr": val_auc_pr})
    wandb.finish()


def main() -> int:
    """Run the HPO study and register the best model."""
    load_dotenv()
    cfg = load_config()

    df = load_features(Path(cfg["paths"]["processed_data"]))
    train_df, test_df = time_based_split(df, float(cfg["split"]["test_fraction"]))
    # Tune on a validation slice carved (time-based) from the training period —
    # never on the test set.
    hpo_train_df, hpo_val_df = time_based_split(train_df, VALIDATION_FRACTION)
    X_tr, y_tr = to_xy(hpo_train_df)
    X_val, y_val = to_xy(hpo_val_df)
    y_val_arr = np.asarray(y_val.to_numpy(), dtype=np.int_)
    spw = compute_scale_pos_weight(y_tr)
    recall_target = float(cfg["thresholds"]["recall_target"])

    mlflow.set_tracking_uri(str(cfg["paths"]["mlflow_tracking_uri"]))
    mlflow.set_experiment(str(cfg["paths"]["experiment_name"]))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        with mlflow.start_run(nested=True):
            model = FraudModel(params=params, scale_pos_weight=spw)
            model.fit(X_tr, y_tr)
            metrics = compute_metrics(y_val_arr, model.predict_proba(X_val), recall_target)
            mlflow.log_params(params)
            mlflow.log_metric("val_auc_pr", metrics["auc_pr"])
            try:
                log_trial_to_wandb(params, metrics["auc_pr"])
            except Exception as exc:
                logger.warning("W&B trial logging skipped: %s", exc)
            return metrics["auc_pr"]

    with mlflow.start_run(run_name="hpo"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=N_TRIALS)
        logger.info("Best validation AUC-PR: %.4f", study.best_value)
        mlflow.log_metric("best_val_auc_pr", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

        # Retrain the best config on the FULL training set; evaluate on test.
        best_params: dict[str, Any] = {
            **study.best_params,
            "eval_metric": "aucpr",
            "random_state": 42,
        }
        X_train, y_train = to_xy(train_df)
        X_test, y_test = to_xy(test_df)
        y_test_arr = np.asarray(y_test.to_numpy(), dtype=np.int_)

        final = FraudModel(params=best_params, scale_pos_weight=compute_scale_pos_weight(y_train))
        final.fit(X_train, y_train)
        test_proba = final.predict_proba(X_test)
        test_metrics = compute_metrics(y_test_arr, test_proba, recall_target)
        logger.info("Test metrics (tuned): %s", {k: round(v, 4) for k, v in test_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        min_auc_pr = float(cfg["thresholds"]["promotion_auc_pr"])
        if not passes_promotion_gate(test_metrics, min_auc_pr):
            logger.error(
                "❌ Gate FAILED: test auc_pr=%.4f < %.2f — not registering.",
                test_metrics["auc_pr"],
                min_auc_pr,
            )
            return 1

        signature = infer_signature(X_test, test_proba)
        mlflow.xgboost.log_model(
            final.model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(3),
            registered_model_name=str(cfg["paths"]["registered_model_name"]),
        )
        logger.info(
            "✅ Registered tuned '%s' (test auc_pr=%.4f).",
            cfg["paths"]["registered_model_name"],
            test_metrics["auc_pr"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
