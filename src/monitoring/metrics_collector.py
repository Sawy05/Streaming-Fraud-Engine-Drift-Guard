"""Prometheus metrics collector for prediction drift, null feature spikes, and P95/P99 latency."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_LATENCY_BUCKETS = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    25.0,
    50.0,
)


class FraudMetricsCollector:
    """Thread-safe Prometheus telemetry collector for the Streaming Fraud Engine."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self._latency_window: deque[float] = deque(maxlen=2000)

        self.inference_latency_histogram = Histogram(
            "fraud_inference_latency_ms",
            "End-to-end fraud inference latency in milliseconds (Feast lookup + ONNX Runtime).",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.latency_p50_gauge = Gauge(
            "fraud_inference_p50_latency_ms",
            "Rolling P50 fraud inference latency in milliseconds.",
            registry=self.registry,
        )
        self.latency_p95_gauge = Gauge(
            "fraud_inference_p95_latency_ms",
            "Rolling P95 fraud inference latency in milliseconds.",
            registry=self.registry,
        )
        self.latency_p99_gauge = Gauge(
            "fraud_inference_p99_latency_ms",
            "Rolling P99 fraud inference latency in milliseconds.",
            registry=self.registry,
        )

        self.predictions_total = Counter(
            "fraud_predictions_total",
            "Total number of scored transactions partitioned by decision outcome.",
            labelnames=("decision",),
            registry=self.registry,
        )

        self.prediction_drift_wasserstein = Gauge(
            "fraud_prediction_drift_wasserstein",
            "Normalized Wasserstein distance of rolling fraud probability distribution vs baseline.",
            registry=self.registry,
        )
        self.prediction_drift_ks_stat = Gauge(
            "fraud_prediction_drift_ks_stat",
            "Kolmogorov-Smirnov test statistic of rolling fraud probability distribution.",
            registry=self.registry,
        )
        self.prediction_drift_ks_pvalue = Gauge(
            "fraud_prediction_drift_ks_pvalue",
            "Kolmogorov-Smirnov p-value of rolling fraud probability distribution.",
            registry=self.registry,
        )

        self.feature_drift_wasserstein = Gauge(
            "fraud_feature_drift_wasserstein",
            "Normalized Wasserstein distance per monitored feature across rolling window.",
            labelnames=("feature",),
            registry=self.registry,
        )
        self.feature_drift_ks_stat = Gauge(
            "fraud_feature_drift_ks_stat",
            "Kolmogorov-Smirnov D-statistic per monitored feature across rolling window.",
            labelnames=("feature",),
            registry=self.registry,
        )
        self.feature_drift_ks_pvalue = Gauge(
            "fraud_feature_drift_ks_pvalue",
            "Kolmogorov-Smirnov p-value per monitored feature across rolling window.",
            labelnames=("feature",),
            registry=self.registry,
        )
        self.feature_drift_detected = Gauge(
            "fraud_feature_drift_detected",
            "Binary indicator (1 = drifted, 0 = stable) per monitored feature.",
            labelnames=("feature",),
            registry=self.registry,
        )

        self.null_feature_spike_ratio = Gauge(
            "fraud_null_feature_spike_ratio",
            "Fraction of null/missing values observed for a feature in the current rolling window.",
            labelnames=("feature",),
            registry=self.registry,
        )
        self.null_feature_events_total = Counter(
            "fraud_null_feature_events_total",
            "Cumulative count of incoming events with missing/null feature values.",
            labelnames=("feature",),
            registry=self.registry,
        )

        self.drift_webhook_alerts_total = Counter(
            "fraud_drift_webhook_alerts_total",
            "Total automated drift alert webhooks triggered.",
            labelnames=("severity",),
            registry=self.registry,
        )

    def record_inference(self, latency_ms: float, decision: str) -> None:
        """Record a single transaction scoring latency and decision outcome."""
        self.inference_latency_histogram.observe(latency_ms)
        self.predictions_total.labels(decision=decision).inc()
        self._latency_window.append(latency_ms)

        if len(self._latency_window) >= 5:
            arr = np.fromiter(self._latency_window, dtype=np.float64)
            p50, p95, p99 = np.percentile(arr, [50, 95, 99])
            self.latency_p50_gauge.set(float(p50))
            self.latency_p95_gauge.set(float(p95))
            self.latency_p99_gauge.set(float(p99))
        else:
            self.latency_p50_gauge.set(latency_ms)
            self.latency_p95_gauge.set(latency_ms)
            self.latency_p99_gauge.set(latency_ms)

    def record_null_feature(self, feature_name: str, count: int = 1) -> None:
        """Increment null/NaN feature counter."""
        if count > 0:
            self.null_feature_events_total.labels(feature=feature_name).inc(count)

    def update_drift_metrics(self, drift_report: dict[str, Any]) -> None:
        """Publish computed KS and Wasserstein drift metrics to Prometheus gauges."""
        pred_drift = drift_report.get("prediction_drift", {})
        if pred_drift:
            self.prediction_drift_wasserstein.set(
                float(pred_drift.get("wasserstein_distance", 0.0))
            )
            self.prediction_drift_ks_stat.set(float(pred_drift.get("ks_statistic", 0.0)))
            self.prediction_drift_ks_pvalue.set(float(pred_drift.get("ks_p_value", 1.0)))

        for feat_name, feat_stats in drift_report.get("feature_drift", {}).items():
            self.feature_drift_wasserstein.labels(feature=feat_name).set(
                float(feat_stats.get("wasserstein_distance", 0.0))
            )
            self.feature_drift_ks_stat.labels(feature=feat_name).set(
                float(feat_stats.get("ks_statistic", 0.0))
            )
            self.feature_drift_ks_pvalue.labels(feature=feat_name).set(
                float(feat_stats.get("ks_p_value", 1.0))
            )
            self.feature_drift_detected.labels(feature=feat_name).set(
                1.0 if feat_stats.get("drift_detected", False) else 0.0
            )
            self.null_feature_spike_ratio.labels(feature=feat_name).set(
                float(feat_stats.get("null_ratio", 0.0))
            )

    def get_latency_summary(self) -> dict[str, float]:
        """Return current rolling P50, P95, and P99 latencies in milliseconds."""
        if not self._latency_window:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0.0}
        arr = np.fromiter(self._latency_window, dtype=np.float64)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        return {
            "p50_ms": round(float(p50), 4),
            "p95_ms": round(float(p95), 4),
            "p99_ms": round(float(p99), 4),
            "count": float(len(arr)),
        }

    def render_prometheus_text(self) -> tuple[bytes, str]:
        """Generate Prometheus exposition payload and content type header."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


metrics_collector = FraudMetricsCollector()
