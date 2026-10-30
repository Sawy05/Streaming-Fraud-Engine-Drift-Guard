FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY feature_repo/ ./feature_repo/

RUN pip install --upgrade pip && pip install .

# Pre-build offline historical dataset, train LightGBM Focal Loss model, and export to ONNX
RUN python -m src.training.train

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
