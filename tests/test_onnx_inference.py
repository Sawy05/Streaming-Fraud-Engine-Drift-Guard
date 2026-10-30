"""Unit and pytest-benchmark latency tests for LightGBM Focal Loss, ONNX Runtime, and FastAPI serving."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.serving.app import app
from src.serving.onnx_engine import ONNXFraudEngine
from src.serving.schemas import TransactionRequest
from src.training.export_onnx import export_lightgbm_to_onnx
from src.training.train import train_fraud_model


@pytest.fixture(scope="module")
def trained_onnx_engine() -> ONNXFraudEngine:
    """Ensure LightGBM Focal Loss model is trained and exported to ONNX before running inference tests."""
    train_meta = train_fraud_model(num_boost_round=50)
    assert train_meta["val_pr_auc"] > 0.70

    export_report = export_lightgbm_to_onnx()
    assert export_report["max_abs_diff_vs_lightgbm"] < 1e-4
    assert export_report["single_inference_p99_ms"] < 15.0

    return ONNXFraudEngine()


def test_onnx_engine_legit_vs_fraud_separation(trained_onnx_engine: ONNXFraudEngine) -> None:
    """Verify ONNX Runtime engine assigns higher fraud probability to anomalous transactions."""
    trained_onnx_engine.online_store.upsert_user_features(
        user_id="usr_00010",
        user_avg_spend_24h=48.0,
        user_txn_count_1h=1,
    )

    legit_req = TransactionRequest(
        transaction_id="txn_legit_001",
        user_id="usr_00010",
        amount=42.50,
        merchant_category="grocery",
        timestamp="2026-10-30T14:30:00+00:00",
        geo_lat=39.82,
        geo_long=-98.58,
        device_fingerprint="dev_trusted_usr_00010",
    )

    trained_onnx_engine.online_store.upsert_user_features(
        user_id="usr_00099",
        user_avg_spend_24h=45.0,
        user_txn_count_1h=14,
    )
    fraud_req = TransactionRequest(
        transaction_id="txn_fraud_001",
        user_id="usr_00099",
        amount=4850.00,
        merchant_category="crypto_exchange",
        timestamp="2026-10-30T03:15:00+00:00",
        geo_lat=-22.90,
        geo_long=135.10,
        device_fingerprint="dev_untrusted_spoof_991",
    )

    legit_resp, _ = trained_onnx_engine.predict(legit_req)
    fraud_resp, _ = trained_onnx_engine.predict(fraud_req)

    assert 0.0 <= legit_resp.fraud_probability <= 1.0
    assert 0.0 <= fraud_resp.fraud_probability <= 1.0
    assert fraud_resp.fraud_probability > legit_resp.fraud_probability
    assert fraud_resp.is_fraud is True
    assert fraud_resp.decision == "BLOCK"


def test_p99_inference_latency_sla_benchmark(
    trained_onnx_engine: ONNXFraudEngine,
    benchmark: Any,
) -> None:
    """Benchmark single-event ONNX Runtime + Redis feature lookup and enforce P99 < 15ms SLA."""
    sample_req = TransactionRequest(
        transaction_id="txn_bench_001",
        user_id="usr_00010",
        amount=129.99,
        merchant_category="electronics",
        timestamp="2026-10-30T15:00:00+00:00",
        geo_lat=38.90,
        geo_long=-97.10,
        device_fingerprint="dev_trusted_usr_00010",
    )

    def _run_single() -> float:
        resp, _ = trained_onnx_engine.predict(sample_req)
        return resp.total_latency_ms

    first_latency = benchmark(_run_single)
    assert first_latency < 15.0

    # Measure empirical P99 across 300 consecutive calls
    latencies = [_run_single() for _ in range(300)]
    p99_ms = float(np.percentile(latencies, 99))
    assert p99_ms < 15.0, f"P99 latency {p99_ms:.3f}ms exceeded 15.0ms SLA!"


def test_fastapi_predict_and_observability_endpoints(trained_onnx_engine: ONNXFraudEngine) -> None:
    """Verify FastAPI /v1/predict, /v1/predict/batch, /health, and /metrics endpoints."""
    with TestClient(app) as client:
        payload = {
            "transaction_id": "txn_api_001",
            "user_id": "usr_00010",
            "amount": 89.50,
            "merchant_category": "retail",
            "timestamp": "2026-10-30T12:00:00+00:00",
            "geo_lat": 39.85,
            "geo_long": -98.55,
            "device_fingerprint": "dev_trusted_usr_00010",
        }
        resp = client.post("/v1/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["transaction_id"] == "txn_api_001"
        assert body["total_latency_ms"] < 15.0
        assert body["decision"] in ("APPROVE", "REVIEW", "BLOCK")

        batch_resp = client.post("/v1/predict/batch", json={"transactions": [payload, payload]})
        assert batch_resp.status_code == 200
        batch_body = batch_resp.json()
        assert batch_body["batch_size"] == 2
        assert len(batch_body["predictions"]) == 2

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["onnx_model_loaded"] is True

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "fraud_inference_latency_ms" in metrics_resp.text
        assert "fraud_predictions_total" in metrics_resp.text
