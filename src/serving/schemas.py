"""Pydantic v2 request and response schemas for real-time fraud scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Incoming real-time financial transaction event payload."""

    transaction_id: str = Field(..., min_length=1, examples=["txn_8a91bc042f109e12"])
    user_id: str = Field(..., min_length=1, examples=["usr_00042"])
    amount: float = Field(..., gt=0.0, le=1_000_000.0, examples=[249.99])
    merchant_category: str = Field(..., min_length=1, examples=["electronics"])
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        examples=["2026-10-30T14:00:00+00:00"],
    )
    geo_lat: float = Field(..., ge=-90.0, le=90.0, examples=[39.8283])
    geo_long: float = Field(..., ge=-180.0, le=180.0, examples=[-98.5795])
    device_fingerprint: str = Field(..., min_length=1, examples=["dev_trusted_usr_00042"])


class OnlineFeaturesSnapshot(BaseModel):
    """Feast / Redis online feature values retrieved during real-time scoring."""

    user_avg_spend_24h: float
    user_txn_count_1h: float
    merchant_risk_score: float
    lookup_latency_ms: float


class FraudPredictionResponse(BaseModel):
    """Low-latency fraud prediction response with latency breakdown and feature snapshot."""

    transaction_id: str
    user_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    threshold_used: float
    inference_latency_ms: float
    total_latency_ms: float
    online_features: OnlineFeaturesSnapshot
    model_version: str = "onnx-lgbm-focal-v1"


class BatchTransactionRequest(BaseModel):
    """Batch request containing multiple transactions for vectorized ONNX scoring."""

    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class BatchFraudPredictionResponse(BaseModel):
    """Vectorized batch fraud prediction output."""

    predictions: list[FraudPredictionResponse]
    batch_size: int
    total_latency_ms: float
    mean_latency_per_event_ms: float


class UserFeatureUpdateRequest(BaseModel):
    """Payload to materialize or update a user's online features in Redis."""

    user_id: str
    user_avg_spend_24h: float = Field(..., ge=0.0)
    user_txn_count_1h: int = Field(..., ge=0)


class HealthCheckResponse(BaseModel):
    """Service readiness, SLA compliance, and dependency health status."""

    status: Literal["healthy", "degraded"]
    onnx_model_loaded: bool
    online_store: dict[str, Any]
    latency_sla_p99_target_ms: float
    latency_metrics: dict[str, float]
    sla_compliant: bool
