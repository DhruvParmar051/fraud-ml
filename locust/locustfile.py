"""Locust load test for the /predict endpoint.

Run (with the server up at localhost:8000):
    locust -f locust/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 60s --headless \
           --html reports/locust_report.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Real materialized destinations (warm Feast features) + the load also mixes in
# random unseen ids (cold start) to avoid Redis cache skew.
WARM_DESTINATIONS = [
    "C1286084959",
    "C1360767589",
    "C665576141",
    "C97730845",
    "C248609774",
    "C2083562754",
    "C2006081398",
    "C1789550256",
    "C1590550415",
    "C1023714065",
]


def _destination() -> str:
    if random.random() < 0.7:
        return random.choice(WARM_DESTINATIONS)
    return f"C{random.randint(1, 9_999_999)}"


def _payload() -> dict[str, object]:
    amount = round(random.uniform(10.0, 500_000.0), 2)
    old_orig = round(random.uniform(0.0, 1_000_000.0), 2)
    return {
        "customer_id": f"C{random.randint(1, 9_999_999)}",
        "destination_id": _destination(),
        "transaction_type": random.choice(["TRANSFER", "CASH_OUT"]),
        "amount": amount,
        "old_balance_orig": old_orig,
        "new_balance_orig": max(0.0, old_orig - amount),
        "old_balance_dest": round(random.uniform(0.0, 200_000.0), 2),
        "new_balance_dest": round(random.uniform(0.0, 200_000.0), 2),
    }


class FraudUser(HttpUser):
    """Simulated client posting transactions to /predict."""

    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        self.client.post("/predict", json=_payload())
