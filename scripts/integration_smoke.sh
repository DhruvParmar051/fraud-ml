#!/usr/bin/env bash
# End-to-end integration smoke (Week 22).
#
# Starts every moving piece, sends some traffic, verifies Prometheus sees
# /predict counters, then tears down. Requires Docker Desktop running and the
# fraud-ml conda env activated.
set -euo pipefail

LOGS="${LOGS_DIR:-/tmp/fraud-integration}"
mkdir -p "$LOGS"
echo "logs -> $LOGS"

PIDS=()
cleanup() {
  echo "--- cleanup ---"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

# helper: wait until a URL responds OK (no foreground sleep loop needed)
wait_http() {
  curl -sf --max-time 60 --retry 60 --retry-delay 1 --retry-all-errors -o /dev/null "$1"
}

echo "--- 1. infra ---"
docker compose -f docker/docker-compose.yml up -d
wait_http http://localhost:5050/health
echo "mlflow ok"

echo "--- 2. serving API ---"
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 > "$LOGS/serving.log" 2>&1 &
PIDS+=($!)
wait_http http://localhost:8000/health
echo "serving ok"

echo "--- 3. streaming consumer ---"
python -m src.data.kafka_consumer > "$LOGS/consumer.log" 2>&1 &
PIDS+=($!)

echo "--- 4. producer (1000 msgs @ 100 tps) ---"
python -m src.data.kafka_producer --tps 100 --limit 1000 > "$LOGS/producer.log" 2>&1

echo "--- 5. light load test (10 users, 30s) ---"
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 10 --spawn-rate 2 --run-time 30s --headless \
       > "$LOGS/locust.log" 2>&1

echo "--- 6. verify Prometheus scraped /predict ---"
COUNT=$(curl -sf "http://localhost:9090/api/v1/query?query=fraud_predictions_total" \
        | python -c "import json,sys; d=json.load(sys.stdin); print(int(float(d['data']['result'][0]['value'][1])) if d['data']['result'] else 0)")
echo "fraud_predictions_total seen by prometheus: $COUNT"
test "$COUNT" -gt 0

echo "--- 7. verify Redis live counters (consumer side) ---"
KEYS=$(docker exec fraud-redis redis-cli --scan --pattern 'dest:*' | wc -l | tr -d ' ')
echo "live dest:* counters in redis: $KEYS"
test "$KEYS" -gt 0

echo "✅ INTEGRATION OK"
