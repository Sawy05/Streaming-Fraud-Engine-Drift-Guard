"""Evidently AI & statistical drift detector (KS test + Wasserstein distance) with webhook alerting."""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
import pandas as pd
from scipy import stats

from src.config import settings
from src.monitoring.metrics_collector import FraudMetricsCollector, metrics_collector

logger = logging.getLogger(__name__)

MONITORED_FEATURES: tuple[str, ...] = (
    "amount",
    "user_avg_spend_24h",
    "user_txn_count_1h",
    "merchant_risk_score",
    "spend_to_avg_ratio",
    "geo_dist_from_center_km",
)


def compute_ks_and_wasserstein(
    reference: np.ndarray,
    current: np.ndarray,
    ks_alpha: float = 0.05,
    wasserstein_threshold: float = 0.15,
) -> dict[str, Any]:
    """Compute two-sample Kolmogorov-Smirnov test and scale-normalized Wasserstein-1 distance."""
    ref_clean = reference[~np.isnan(reference)].astype(np.float64)
    cur_clean = current[~np.isnan(current)].astype(np.float64)

    null_ratio = float(np.isnan(current).sum()) / float(len(current)) if len(current) > 0 else 0.0

    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return {
            "ks_statistic": 0.0,
            "ks_p_value": 1.0,
            "wasserstein_distance": 0.0,
            "raw_wasserstein": 0.0,
            "null_ratio": round(null_ratio, 5),
            "drift_detected": False,
        }

    ks_res = stats.ks_2samp(ref_clean, cur_clean)
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    raw_w = float(stats.wasserstein_distance(ref_clean, cur_clean))
    ref_std = float(np.std(ref_clean))
    norm_denom = ref_std if ref_std > 1e-6 else max(float(np.abs(np.mean(ref_clean))), 1e-6)
    norm_wasserstein = raw_w / norm_denom

    drift_detected = bool(
        (ks_pval < ks_alpha and norm_wasserstein >= wasserstein_threshold)
        or norm_wasserstein >= (wasserstein_threshold * 2.0)
    )

    return {
        "ks_statistic": round(ks_stat, 5),
        "ks_p_value": round(ks_pval, 6),
        "wasserstein_distance": round(norm_wasserstein, 5),
        "raw_wasserstein": round(raw_w, 5),
        "null_ratio": round(null_ratio, 5),
        "drift_detected": drift_detected,
    }


class RollingDriftDetector:
    """Continuous drift monitor comparing rolling live transaction windows against baseline training data."""

    def __init__(
        self,
        reference_df: pd.DataFrame | None = None,
        window_size: int | None = None,
        ks_alpha: float | None = None,
        wasserstein_threshold: float | None = None,
        null_spike_threshold: float | None = None,
        webhook_url: str | None = None,
        collector: FraudMetricsCollector | None = None,
    ) -> None:
        self.window_size = window_size or settings.drift_window_size
        self.ks_alpha = ks_alpha if ks_alpha is not None else settings.drift_ks_alpha
        self.wasserstein_threshold = (
            wasserstein_threshold
            if wasserstein_threshold is not None
            else settings.drift_wasserstein_threshold
        )
        self.null_spike_threshold = (
            null_spike_threshold
            if null_spike_threshold is not None
            else settings.null_spike_threshold
        )
        self.webhook_url = webhook_url if webhook_url is not None else settings.alert_webhook_url
        self.collector = collector or metrics_collector
        self._rolling_window: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._webhook_history: list[dict[str, Any]] = []
        self._reference_arrays: dict[str, np.ndarray] = self._load_reference(reference_df)

    def _load_reference(self, reference_df: pd.DataFrame | None) -> dict[str, np.ndarray]:
        """Load baseline training distributions from DataFrame, Parquet, or synthetic fallback."""
        if reference_df is None and settings.reference_baseline_path.exists():
            try:
                reference_df = pd.read_parquet(settings.reference_baseline_path)
            except Exception as exc:
                logger.warning("Failed to read reference baseline parquet: %s", exc)

        if reference_df is not None and not reference_df.empty:
            ref_map: dict[str, np.ndarray] = {}
            for col in MONITORED_FEATURES:
                if col in reference_df.columns:
                    ref_map[col] = reference_df[col].to_numpy(dtype=np.float64)
            if "fraud_probability" in reference_df.columns:
                ref_map["fraud_probability"] = reference_df["fraud_probability"].to_numpy(
                    dtype=np.float64
                )
            elif "is_fraud" in reference_df.columns:
                rng = np.random.default_rng(42)
                labels = reference_df["is_fraud"].to_numpy(dtype=np.float64)
                ref_map["fraud_probability"] = np.clip(
                    np.where(
                        labels == 1.0,
                        rng.beta(5.0, 2.0, size=len(labels)),
                        rng.beta(1.2, 18.0, size=len(labels)),
                    ),
                    0.0,
                    1.0,
                )
            if ref_map:
                return ref_map

        # Deterministic synthetic baseline fallback
        rng = np.random.default_rng(42)
        n_ref = 1000
        amount = rng.lognormal(mean=3.85, sigma=0.65, size=n_ref)
        avg_spend = rng.lognormal(mean=3.9, sigma=0.45, size=n_ref)
        return {
            "amount": amount,
            "user_avg_spend_24h": avg_spend,
            "user_txn_count_1h": rng.poisson(lam=1.8, size=n_ref).astype(np.float64) + 1.0,
            "merchant_risk_score": rng.choice([0.03, 0.08, 0.22, 0.35, 0.78], size=n_ref),
            "spend_to_avg_ratio": amount / (avg_spend + 1.0),
            "geo_dist_from_center_km": np.abs(rng.normal(450.0, 220.0, size=n_ref)),
            "fraud_probability": rng.beta(1.2, 18.0, size=n_ref),
        }

    def observe_event(self, payload: dict[str, float]) -> None:
        """Append an incoming transaction's feature & prediction snapshot to the rolling window."""
        for feat in MONITORED_FEATURES:
            val = payload.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                self.collector.record_null_feature(feat, 1)
        self._rolling_window.append(payload)

    def observe_batch(self, df: pd.DataFrame) -> None:
        """Ingest a DataFrame batch of scored transactions into the rolling window."""
        for record in df.to_dict(orient="records"):
            self.observe_event(record)

    def run_evidently_drift_summary(
        self,
        current_df: pd.DataFrame,
        full_evidently_report: bool = False,
    ) -> dict[str, Any]:
        """Run Evidently AI drift evaluation alongside KS & Wasserstein statistical tests."""
        import importlib.util

        evidently_installed = importlib.util.find_spec("evidently") is not None
        cols = [
            c for c in MONITORED_FEATURES if c in current_df.columns and c in self._reference_arrays
        ]
        if not cols or len(current_df) < 5:
            return {
                "evidently_enabled": evidently_installed,
                "drifted_columns": [],
                "share_of_drifted_columns": 0.0,
            }

        ref_df = pd.DataFrame({c: self._reference_arrays[c][: len(current_df)] for c in cols})
        cur_sub = current_df[cols].copy()

        if full_evidently_report and evidently_installed:
            try:
                from evidently.metrics import DataDriftTable  # type: ignore[import-not-found]
                from evidently.report import Report  # type: ignore[import-not-found]

                report = Report(metrics=[DataDriftTable()])
                report.run(reference_data=ref_df, current_data=cur_sub)
                res_dict = report.as_dict()
                drift_res = res_dict["metrics"][0]["result"]
                return {
                    "evidently_enabled": True,
                    "engine": "evidently.report.Report",
                    "dataset_drift": bool(drift_res.get("dataset_drift", False)),
                    "share_of_drifted_columns": float(
                        drift_res.get("share_of_drifted_columns", 0.0)
                    ),
                }
            except Exception as exc:
                logger.debug("Fallback from full Evidently report: %s", exc)

        drifted = []
        for c in cols:
            res = compute_ks_and_wasserstein(
                self._reference_arrays[c],
                cur_sub[c].to_numpy(dtype=np.float64),
                ks_alpha=self.ks_alpha,
                wasserstein_threshold=self.wasserstein_threshold,
            )
            if res["drift_detected"]:
                drifted.append(c)
        share = len(drifted) / max(1, len(cols))
        return {
            "evidently_enabled": evidently_installed,
            "engine": "evidently_ks_wasserstein_hybrid",
            "dataset_drift": share >= 0.30,
            "drifted_columns": drifted,
            "share_of_drifted_columns": round(share, 4),
        }

    def evaluate_window(self, trigger_webhook: bool = True) -> dict[str, Any]:
        """Evaluate KS tests, Wasserstein distances, and null spikes across the current rolling window."""
        if not self._rolling_window:
            return {
                "window_size": 0,
                "overall_drift_detected": False,
                "null_spike_detected": False,
                "drifted_features": [],
                "null_spiked_features": [],
                "feature_drift": {},
                "prediction_drift": {},
                "timestamp": datetime.now(UTC).isoformat(),
            }

        window_df = pd.DataFrame(list(self._rolling_window))
        feature_drift: dict[str, dict[str, Any]] = {}
        drifted_features: list[str] = []
        null_spiked_features: list[str] = []

        for feat in MONITORED_FEATURES:
            if feat not in window_df.columns or feat not in self._reference_arrays:
                continue
            cur_vals = window_df[feat].to_numpy(dtype=np.float64)
            stats_dict = compute_ks_and_wasserstein(
                self._reference_arrays[feat],
                cur_vals,
                ks_alpha=self.ks_alpha,
                wasserstein_threshold=self.wasserstein_threshold,
            )
            feature_drift[feat] = stats_dict
            if stats_dict["drift_detected"]:
                drifted_features.append(feat)
            if stats_dict["null_ratio"] >= self.null_spike_threshold:
                null_spiked_features.append(feat)

        prediction_drift: dict[str, Any] = {}
        if (
            "fraud_probability" in window_df.columns
            and "fraud_probability" in self._reference_arrays
        ):
            prediction_drift = compute_ks_and_wasserstein(
                self._reference_arrays["fraud_probability"],
                window_df["fraud_probability"].to_numpy(dtype=np.float64),
                ks_alpha=self.ks_alpha,
                wasserstein_threshold=self.wasserstein_threshold,
            )

        evidently_summary = self.run_evidently_drift_summary(window_df)
        overall_drift = bool(drifted_features or prediction_drift.get("drift_detected", False))
        null_spike = bool(null_spiked_features)

        report: dict[str, Any] = {
            "window_size": len(window_df),
            "overall_drift_detected": overall_drift,
            "null_spike_detected": null_spike,
            "drifted_features": drifted_features,
            "null_spiked_features": null_spiked_features,
            "feature_drift": feature_drift,
            "prediction_drift": prediction_drift,
            "evidently_summary": evidently_summary,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.collector.update_drift_metrics(report)

        if trigger_webhook and (overall_drift or null_spike):
            self.dispatch_webhook_alert(report)

        return report

    def dispatch_webhook_alert(self, drift_report: dict[str, Any]) -> dict[str, Any]:
        """Trigger automated alert webhook when feature drift or null spikes exceed thresholds."""
        severity = (
            "CRITICAL"
            if len(drift_report.get("drifted_features", [])) >= 2
            or drift_report.get("null_spike_detected", False)
            else "WARNING"
        )
        alert_payload = {
            "alert_type": "MLOPS_FEATURE_AND_PREDICTION_DRIFT",
            "severity": severity,
            "service": "streaming-fraud-mlops",
            "drifted_features": drift_report.get("drifted_features", []),
            "null_spiked_features": drift_report.get("null_spiked_features", []),
            "prediction_drift": drift_report.get("prediction_drift", {}),
            "window_size": drift_report.get("window_size", 0),
            "timestamp": drift_report.get("timestamp"),
        }

        self.collector.drift_webhook_alerts_total.labels(severity=severity).inc()
        delivered = False
        status_code: int | None = None

        if self.webhook_url:
            try:
                with httpx.Client(timeout=settings.alert_webhook_timeout_sec) as client:
                    response = client.post(self.webhook_url, json=alert_payload)
                    status_code = response.status_code
                    delivered = 200 <= response.status_code < 300
            except Exception as exc:
                logger.warning("Alert webhook delivery to %s failed: %s", self.webhook_url, exc)

        record = {
            **alert_payload,
            "webhook_url": self.webhook_url,
            "delivered": delivered,
            "status_code": status_code,
        }
        self._webhook_history.append(record)
        return record

    @property
    def webhook_history(self) -> list[dict[str, Any]]:
        """Return list of dispatched drift alert webhooks."""
        return list(self._webhook_history)
