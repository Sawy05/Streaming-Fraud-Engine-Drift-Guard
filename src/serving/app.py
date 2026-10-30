"""FastAPI low-latency inference service with ONNX Runtime, asynchronous drift logging, and Prometheus metrics."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.monitoring.drift_detector import RollingDriftDetector
from src.monitoring.metrics_collector import metrics_collector
from src.serving.onnx_engine import ONNXFraudEngine
from src.serving.schemas import (
    BatchFraudPredictionResponse,
    BatchTransactionRequest,
    FraudPredictionResponse,
    HealthCheckResponse,
    TransactionRequest,
    UserFeatureUpdateRequest,
)

logger = logging.getLogger(__name__)


class ServiceState:
    """Singleton application state holding ONNX engine, drift monitor, and webhook sink."""

    def __init__(self) -> None:
        self.engine: ONNXFraudEngine | None = None
        self.drift_detector: RollingDriftDetector | None = None
        self.received_webhook_alerts: list[dict[str, Any]] = []
        self._scored_since_last_eval: int = 0

    def get_engine(self) -> ONNXFraudEngine:
        if self.engine is None:
            self.engine = ONNXFraudEngine()
        return self.engine

    def get_drift_detector(self) -> RollingDriftDetector:
        if self.drift_detector is None:
            self.drift_detector = RollingDriftDetector(collector=metrics_collector)
        return self.drift_detector

    def record_payload_async(self, monitoring_payload: dict[str, float]) -> None:
        """Asynchronously log incoming payload distribution and periodically evaluate drift."""
        detector = self.get_drift_detector()
        detector.observe_event(monitoring_payload)
        self._scored_since_last_eval += 1
        if self._scored_since_last_eval >= 100:
            self._scored_since_last_eval = 0
            detector.evaluate_window(trigger_webhook=True)


state = ServiceState()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm up ONNX Runtime session and baseline drift detector on startup."""
    state.get_engine()
    state.get_drift_detector()
    logger.info("Streaming Fraud Engine initialized and warmed up.")
    yield


app = FastAPI(
    title="Streaming Fraud Engine & Drift Guard",
    description=(
        "Real-time financial transaction fraud detection powered by Feast/Redis online features, "
        "LightGBM Focal Loss exported to ONNX Runtime (<15ms P99 SLA), and Evidently AI drift guard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse, tags=["Observability"])
async def health_check() -> HealthCheckResponse:
    """Return service readiness, Redis online store connectivity, and P99 SLA compliance."""
    engine = state.get_engine()
    store_status = engine.online_store.health_check()
    latency_summary = metrics_collector.get_latency_summary()
    p99_ms = latency_summary["p99_ms"]
    sla_ok = p99_ms <= settings.sla_p99_latency_ms

    return HealthCheckResponse(
        status="healthy" if (engine.is_loaded and store_status["healthy"]) else "degraded",
        onnx_model_loaded=engine.is_loaded,
        online_store=store_status,
        latency_sla_p99_target_ms=settings.sla_p99_latency_ms,
        latency_metrics=latency_summary,
        sla_compliant=sla_ok,
    )


@app.post("/v1/predict", response_model=FraudPredictionResponse, tags=["Inference"])
async def predict_transaction(
    request: TransactionRequest,
    background_tasks: BackgroundTasks,
) -> FraudPredictionResponse:
    """Score a single real-time transaction and asynchronously log payload distribution for drift monitoring."""
    engine = state.get_engine()
    response, monitoring_payload = engine.predict(request)

    metrics_collector.record_inference(
        latency_ms=response.total_latency_ms,
        decision=response.decision,
    )
    background_tasks.add_task(state.record_payload_async, monitoring_payload)
    return response


@app.post("/v1/predict/batch", response_model=BatchFraudPredictionResponse, tags=["Inference"])
async def predict_transactions_batch(
    request: BatchTransactionRequest,
    background_tasks: BackgroundTasks,
) -> BatchFraudPredictionResponse:
    """Score a batch of transactions using pipelined Redis lookups and vectorized ONNX inference."""
    engine = state.get_engine()
    batch_response, monitoring_rows = engine.predict_batch(request.transactions)

    for pred in batch_response.predictions:
        metrics_collector.record_inference(
            latency_ms=pred.total_latency_ms,
            decision=pred.decision,
        )
    for row in monitoring_rows:
        background_tasks.add_task(state.record_payload_async, row)

    return batch_response


@app.post("/v1/features/user", tags=["Feature Store"])
async def update_user_online_features(request: UserFeatureUpdateRequest) -> dict[str, Any]:
    """Upsert a user's rolling spend and transaction count in the Redis online feature store."""
    engine = state.get_engine()
    engine.online_store.upsert_user_features(
        user_id=request.user_id,
        user_avg_spend_24h=request.user_avg_spend_24h,
        user_txn_count_1h=request.user_txn_count_1h,
    )
    return {
        "status": "updated",
        "user_id": request.user_id,
        "user_avg_spend_24h": request.user_avg_spend_24h,
        "user_txn_count_1h": request.user_txn_count_1h,
    }


@app.get("/v1/drift/status", tags=["Observability"])
async def get_drift_status(trigger_webhook: bool = False) -> dict[str, Any]:
    """Compute Kolmogorov-Smirnov (KS) and Wasserstein drift report across the rolling transaction window."""
    detector = state.get_drift_detector()
    return detector.evaluate_window(trigger_webhook=trigger_webhook)


@app.post("/v1/drift/webhook-sink", tags=["Observability"])
async def receive_drift_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """Built-in webhook receiver sink for automated drift alerts."""
    state.received_webhook_alerts.append(payload)
    logger.warning("Drift Alert Webhook Received: %s", payload)
    return {"status": "acknowledged", "received_alerts_count": len(state.received_webhook_alerts)}


@app.get("/metrics", tags=["Observability"])
async def prometheus_metrics() -> Response:
    """Expose Prometheus telemetry for prediction drift, null spikes, and P50/P95/P99 latency."""
    payload_bytes, content_type = metrics_collector.render_prometheus_text()
    return Response(content=payload_bytes, media_type=content_type)
