"""Replay PaySim transactions to Redpanda (Week 13).

Usage:
    python -m src.data.kafka_producer --tps 100
    python -m src.data.kafka_producer --tps 1000 --limit 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TOPIC = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions.raw")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
N_PARTITIONS = 3


def _json_default(o: object) -> int | float | str:
    """Make numpy scalars JSON-serializable."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def ensure_topic(bootstrap: str, topic: str, partitions: int) -> None:
    """Create the topic if it doesn't already exist."""
    admin = AdminClient({"bootstrap.servers": bootstrap})
    if topic in admin.list_topics(timeout=10).topics:
        logger.info("Topic '%s' already exists.", topic)
        return
    for _, fut in admin.create_topics(
        [NewTopic(topic, num_partitions=partitions, replication_factor=1)]
    ).items():
        fut.result()
    logger.info("Created topic '%s' with %d partitions.", topic, partitions)


def main() -> int:
    """Replay the PaySim CSV to Redpanda at a configurable rate."""
    parser = argparse.ArgumentParser(description="Replay PaySim to Redpanda.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/paysim.csv"))
    parser.add_argument("--tps", type=int, default=100, help="Messages per second.")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (for testing).")
    args = parser.parse_args()

    ensure_topic(BOOTSTRAP, TOPIC, N_PARTITIONS)
    producer = Producer({"bootstrap.servers": BOOTSTRAP})
    df = pd.read_csv(args.input, nrows=args.limit)
    interval = 1.0 / args.tps

    sent = 0
    win_start = time.time()
    win_count = 0
    for row in df.to_dict(orient="records"):
        row["produced_at"] = datetime.now(UTC).isoformat()
        producer.produce(
            TOPIC, key=str(row["nameDest"]), value=json.dumps(row, default=_json_default)
        )
        producer.poll(0)
        sent += 1
        win_count += 1
        now = time.time()
        if now - win_start >= 10:
            logger.info("rate %.0f msg/s, total produced %d", win_count / (now - win_start), sent)
            win_start, win_count = now, 0
        time.sleep(interval)

    producer.flush()
    logger.info("Done. Total produced: %d", sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
