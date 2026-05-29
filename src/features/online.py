"""Real-time online feature retrieval from Feast/Redis (Week 12).

Usage (after `feast materialize ...`):
    from src.features.online import get_online_features
    get_online_features("C1286084959")
"""

from __future__ import annotations

import logging

from feast import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_PATH = "src/features/feature_repo"
WINDOW_FEATURES = [
    "dest_tx_count_1h",
    "dest_tx_count_24h",
    "dest_tx_count_7d",
    "dest_amount_sum_24h",
    "dest_amount_avg_24h",
]
FEATURE_REFS = [f"destination_stats:{f}" for f in WINDOW_FEATURES]

_store: FeatureStore | None = None


def _get_store() -> FeatureStore:
    """Return a cached FeatureStore (created once, reused per process)."""
    global _store
    if _store is None:
        _store = FeatureStore(repo_path=REPO_PATH)
    return _store


def get_online_features(destination_id: str) -> dict[str, float]:
    """Fetch the latest stored window features for a destination from Redis.

    Unknown destinations (cold start) default to 0.0 — no observed history.
    """
    result = (
        _get_store()
        .get_online_features(
            features=FEATURE_REFS,
            entity_rows=[{"nameDest": destination_id}],
        )
        .to_dict()
    )
    return {
        k: (float(v[0]) if v[0] is not None else 0.0) for k, v in result.items() if k != "nameDest"
    }
