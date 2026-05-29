"""Unit tests for serving feature assembly (no infra required)."""

from __future__ import annotations

from src.serving.predict import build_features
from src.serving.schemas import TransactionRequest
from src.training.train import FEATURE_COLUMNS

WINDOW = {
    "dest_tx_count_1h": 0.0,
    "dest_tx_count_24h": 0.0,
    "dest_tx_count_7d": 0.0,
    "dest_amount_sum_24h": 0.0,
    "dest_amount_avg_24h": 0.0,
}


def _req(**kw: object) -> TransactionRequest:
    base = {
        "customer_id": "C1",
        "destination_id": "D1",
        "transaction_type": "TRANSFER",
        "amount": 200.0,
        "old_balance_orig": 1000.0,
        "new_balance_orig": 800.0,
        "old_balance_dest": 500.0,
        "new_balance_dest": 700.0,
    }
    base.update(kw)
    return TransactionRequest(**base)  # type: ignore[arg-type]


def test_columns_match_training_order() -> None:
    df = build_features(_req(), WINDOW)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_derived_features() -> None:
    row = build_features(_req(), WINDOW).iloc[0]
    assert row["balance_delta"] == -200.0
    assert row["errorBalanceOrig"] == 0.0
    assert row["errorBalanceDest"] == 0.0
    assert row["is_transfer"] == 1
    assert row["is_round_amount"] == 1


def test_cold_start_window_defaults_to_zero() -> None:
    row = build_features(_req(), {}).iloc[0]
    assert row["dest_tx_count_24h"] == 0.0
