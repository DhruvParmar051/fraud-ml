# syntax=docker/dockerfile:1
# ---- builder: install deps into an isolated prefix ----
FROM python:3.11-slim AS builder
WORKDIR /build
COPY src/serving/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- final: slim runtime image ----
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY configs/ ./configs/
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
