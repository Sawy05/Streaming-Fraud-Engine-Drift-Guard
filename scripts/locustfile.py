"""Locust load testing script for real-time ONNX fraud inference SLA verification (<15ms P99)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from locust import HttpUser, between, task


class FraudInferenceLoadUser(HttpUser):
    """Simulated payment gateway streaming transactions to the FastAPI ONNX service."""

    wait_time = between(0.005, 0.02)

    @task(10)
    def score_single_transaction(self) -> None:
        payload = {
            "transaction_id": f"txn_{uuid.uuid4().hex[:14]}",
            "user_id": "usr_00042",
            "amount": 119.95,
            "merchant_category": "electronics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "geo_lat": 39.8283,
            "geo_long": -98.5795,
            "device_fingerprint": "dev_trusted_usr_00042",
        }
        with self.client.post(
            "/v1/predict",
            json=payload,
            catch_response=True,
            name="/v1/predict",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            else:
                data = resp.json()
                if data.get("total_latency_ms", 999.0) > 15.0:
                    resp.failure(f"SLA breach: {data.get('total_latency_ms')}ms > 15.0ms")

    @task(1)
    def check_drift_status(self) -> None:
        self.client.get("/v1/drift/status", name="/v1/drift/status")
