# ── Variables ────────────────────────────────────────────
COMPOSE       := docker compose -f docker/docker-compose.yml
RAW_DATA      := data/raw/paysim.csv
PROCESSED_DIR := data/processed/

.PHONY: setup train serve test lint format docker-up docker-down docker-logs load-test drift-report retrain-check

# ── Environment ──────────────────────────────────────────
setup:
	conda env update -f environment.yml --prune
	$(COMPOSE) up -d
	pre-commit install

# ── Pipeline ─────────────────────────────────────────────
train:
	python -m src.data.validate --input $(RAW_DATA)
	python -m src.data.etl --input $(RAW_DATA) --output $(PROCESSED_DIR)
	python -m src.training.hpo

serve:
	uvicorn src.serving.main:app --reload --host 0.0.0.0 --port 8000

# ── Code quality ─────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# ── Infrastructure ───────────────────────────────────────
docker-up:
	$(COMPOSE) up -d

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f

# ── Monitoring / load testing ────────────────────────────
load-test:
	locust -f locust/locustfile.py --host http://localhost:8000 \
	       --users 50 --spawn-rate 5 --run-time 60s --headless \
	       --html reports/locust_report.html

drift-report:
	python -m src.monitoring.drift \
	       --reference data/reference/training_features.parquet \
	       --current data/predictions/recent.parquet

retrain-check:
	python -m src.monitoring.retrain_trigger --report reports/drift_latest.json
