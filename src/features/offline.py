"""Point-in-time-correct training features via Feast (Week 11).

Usage:
    python -m src.features.offline

Retrieves training features through Feast's point-in-time joins and retrains the
model, confirming it reproduces the baseline AUC-PR via the leakage-safe path.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from src.training.evaluate import compute_metrics
from src.training.model import FraudModel
from src.training.train import (
    FEATURE_COLUMNS,
    LABEL,
    compute_scale_pos_weight,
    load_config,
    load_features,
    time_based_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_PATH = "src/features/feature_repo"
BASE_DATE = pd.Timestamp("2024-01-01", tz="UTC")
WINDOW_FEATURES = [
    "dest_tx_count_1h",
    "dest_tx_count_24h",
    "dest_tx_count_7d",
    "dest_amount_sum_24h",
    "dest_amount_avg_24h",
]
FEATURE_REFS = [f"destination_stats:{f}" for f in WINDOW_FEATURES]
# Features the model uses that are NOT stored in Feast — computed at request time
# and carried through the entity_df.
REQUEST_FEATURES = [c for c in FEATURE_COLUMNS if c not in WINDOW_FEATURES]


def build_entity_df(df: pd.DataFrame) -> pd.DataFrame:
    """entity_df = join key + event_timestamp + label + request-time features."""
    out = df.copy()
    out["event_timestamp"] = BASE_DATE + pd.to_timedelta(out["step"], unit="h")
    return out[["nameDest", "event_timestamp", LABEL, *REQUEST_FEATURES]]


def get_training_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time-join the stored window features onto the entity_df."""
    store = FeatureStore(repo_path=REPO_PATH)
    return store.get_historical_features(entity_df=entity_df, features=FEATURE_REFS).to_df()


def main() -> int:
    """Retrieve PIT features via Feast, retrain, and report test metrics."""
    cfg = load_config()
    df = load_features(Path(cfg["paths"]["processed_data"]))
    train_df, test_df = time_based_split(df, float(cfg["split"]["test_fraction"]))

    logger.info("Retrieving point-in-time features via Feast ...")
    train_feat = get_training_features(build_entity_df(train_df))
    test_feat = get_training_features(build_entity_df(test_df))

    X_train, y_train = train_feat[FEATURE_COLUMNS], train_feat[LABEL]
    y_test = np.asarray(test_feat[LABEL].to_numpy(), dtype=np.int_)

    model = FraudModel(params=cfg["model"], scale_pos_weight=compute_scale_pos_weight(y_train))
    model.fit(X_train, y_train)
    metrics = compute_metrics(
        y_test,
        model.predict_proba(test_feat[FEATURE_COLUMNS]),
        float(cfg["thresholds"]["recall_target"]),
    )
    logger.info(
        "Feast-served retrain — test metrics: %s", {k: round(v, 4) for k, v in metrics.items()}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
