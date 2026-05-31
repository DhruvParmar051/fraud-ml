"""FastAPI serving app.

Run:
    uvicorn src.serving.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.serving import metrics, predict
from src.serving.schemas import PredictionResponse, TransactionRequest


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model + SHAP explainer once, at startup."""
    predict.load_predictor()
    metrics.MODEL_VERSION_INFO.labels(version=predict._model_version).set(1)
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns ``status`` and the loaded ``model_stage``."""
    return {"status": "healthy", "model_stage": predict.MODEL_STAGE}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: TransactionRequest) -> PredictionResponse:
    """Score a single transaction and record Prometheus metrics.

    Args:
        req: Incoming transaction payload (see ``TransactionRequest``).

    Returns:
        Fraud probability, decision label, top SHAP contributions, and latency.
    """
    resp = predict.predict(req)
    metrics.FRAUD_PREDICTIONS.labels(decision=resp.decision).inc()
    metrics.PREDICTION_LATENCY.observe(resp.latency_ms / 1000.0)
    return resp
