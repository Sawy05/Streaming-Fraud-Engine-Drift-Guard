"""Unit tests for Evidently AI, Kolmogorov-Smirnov (KS), Wasserstein drift detection, and alert webhooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.monitoring.drift_detector import (
    RollingDriftDetector,
    compute_ks_and_wasserstein,
)
from src.monitoring.metrics_collector import FraudMetricsCollector


def test_ks_and_wasserstein_stationary_vs_drifted() -> None:
    """Verify KS test and normalized Wasserstein distance separate stationary from shifted distributions."""
    rng = np.random.default_rng(101)
    ref = rng.normal(loc=100.0, scale=15.0, size=600)
    stationary = rng.normal(loc=100.5, scale=15.0, size=600)
    shifted = rng.normal(loc=210.0, scale=35.0, size=600)

    stat_res = compute_ks_and_wasserstein(
        reference=ref,
        current=stationary,
        ks_alpha=0.05,
        wasserstein_threshold=0.18,
    )
    assert stat_res["drift_detected"] is False
    assert stat_res["wasserstein_distance"] < 0.18

    shift_res = compute_ks_and_wasserstein(
        reference=ref,
        current=shifted,
        ks_alpha=0.05,
        wasserstein_threshold=0.18,
    )
    assert shift_res["drift_detected"] is True
    assert shift_res["ks_p_value"] < 0.001
    assert shift_res["wasserstein_distance"] > 2.0


def test_rolling_drift_detector_null_spikes_and_webhooks() -> None:
    """Verify RollingDriftDetector detects feature drift, null spikes, updates Prometheus, and fires webhooks."""
    rng = np.random.default_rng(2026)
    n_ref = 400
    ref_df = pd.DataFrame(
        {
            "amount": rng.lognormal(mean=3.8, sigma=0.5, size=n_ref),
            "user_avg_spend_24h": rng.lognormal(mean=3.8, sigma=0.4, size=n_ref),
            "user_txn_count_1h": rng.poisson(lam=2.0, size=n_ref).astype(float),
            "merchant_risk_score": rng.uniform(0.05, 0.25, size=n_ref),
            "spend_to_avg_ratio": rng.uniform(0.5, 1.5, size=n_ref),
            "geo_dist_from_center_km": rng.normal(350.0, 100.0, size=n_ref),
            "fraud_probability": rng.beta(1.2, 18.0, size=n_ref),
        }
    )

    isolated_collector = FraudMetricsCollector()
    detector = RollingDriftDetector(
        reference_df=ref_df,
        window_size=150,
        ks_alpha=0.05,
        wasserstein_threshold=0.20,
        null_spike_threshold=0.05,
        webhook_url="",  # In-memory webhook recording without external network dependence
        collector=isolated_collector,
    )

    # Ingest 120 severely drifted records with 10% null spike on user_avg_spend_24h
    for i in range(120):
        detector.observe_event(
            {
                "amount": float(rng.lognormal(mean=7.2, sigma=0.6)),
                "user_avg_spend_24h": float("nan") if i < 15 else 60.0,
                "user_txn_count_1h": 18.0,
                "merchant_risk_score": 0.85,
                "spend_to_avg_ratio": 15.0,
                "geo_dist_from_center_km": 9200.0,
                "fraud_probability": 0.91,
            }
        )

    report = detector.evaluate_window(trigger_webhook=True)

    assert report["overall_drift_detected"] is True
    assert report["null_spike_detected"] is True
    assert "amount" in report["drifted_features"]
    assert "user_avg_spend_24h" in report["null_spiked_features"]
    assert report["prediction_drift"]["drift_detected"] is True
    assert report["evidently_summary"]["evidently_enabled"] is True

    # Verify webhook alert was recorded and Prometheus metrics updated
    assert len(detector.webhook_history) == 1
    assert detector.webhook_history[0]["severity"] == "CRITICAL"

    prom_bytes, _ = isolated_collector.render_prometheus_text()
    prom_text = prom_bytes.decode("utf-8")
    assert "fraud_prediction_drift_wasserstein" in prom_text
    assert "fraud_null_feature_spike_ratio" in prom_text
    assert 'fraud_drift_webhook_alerts_total{severity="CRITICAL"} 1.0' in prom_text
