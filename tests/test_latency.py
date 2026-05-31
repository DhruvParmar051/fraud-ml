"""Latency regression test — full end-to-end latency is covered by the Locust
load test (needs the model + Redis running)."""

import pytest


@pytest.mark.skip(reason="End-to-end P99 latency is measured by Locust.")
def test_predict_p99_under_100ms() -> None:
    pass
