"""Streaming consumer: maintain real-time destination counters in Redis.

Usage:
    python -m src.data.kafka_consumer            # run continuously
    python -m src.data.kafka_consumer --max-messages 5000   # for testing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import redis
from confluent_kafka import Consumer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TOPIC = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions.raw")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-feature-consumer")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

TTL_1H = 3600
TTL_24H = 86400


def update_counters(r: redis.Redis, tx: dict[str, object]) -> None:
    """Increment rolling per-destination counters with TTL-based windows."""
    dest = str(tx["nameDest"])
    amount = float(tx["amount"])  # type: ignore[arg-type]
    k1 = f"dest:{dest}:tx_count_1h"
    k24 = f"dest:{dest}:tx_count_24h"
    ks = f"dest:{dest}:amount_sum_24h"
    pipe = r.pipeline()
    pipe.incr(k1)
    pipe.expire(k1, TTL_1H)
    pipe.incr(k24)
    pipe.expire(k24, TTL_24H)
    pipe.incrbyfloat(ks, amount)
    pipe.expire(ks, TTL_24H)
    pipe.execute()


def main() -> int:
    """Consume transactions and update Redis counters until stopped."""
    parser = argparse.ArgumentParser(description="Consume transactions, update Redis counters.")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after N (testing).")
    args = parser.parse_args()

    consumer = Consumer(
        {"bootstrap.servers": BOOTSTRAP, "group.id": GROUP, "auto.offset.reset": "earliest"}
    )
    consumer.subscribe([TOPIC])
    r = redis.from_url(REDIS_URL, decode_responses=True)

    processed = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                if args.max_messages is not None:
                    break
                continue
            if msg.error():
                logger.error("consume error: %s", msg.error())
                continue
            update_counters(r, json.loads(msg.value()))
            processed += 1
            if processed % 1000 == 0:
                logger.info("processed %d messages", processed)
            if args.max_messages is not None and processed >= args.max_messages:
                break
    finally:
        consumer.close()
    logger.info("Done. Processed %d messages.", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
