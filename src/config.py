"""Centralized configuration for the Streaming Fraud Engine & Drift Guard pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

MERCHANT_CATEGORIES: Final[tuple[str, ...]] = (
    "retail",
    "grocery",
    "travel",
    "electronics",
    "crypto_exchange",
    "luxury_goods",
    "gaming",
    "wire_transfer",
    "dining",
    "utilities",
)

MERCHANT_CATEGORY_TO_CODE: Final[dict[str, int]] = {
    cat: idx for idx, cat in enumerate(MERCHANT_CATEGORIES)
}

MERCHANT_DEFAULT_RISK_SCORES: Final[dict[str, float]] = {
    "retail": 0.08,
    "grocery": 0.03,
    "travel": 0.22,
    "electronics": 0.35,
    "crypto_exchange": 0.78,
    "luxury_goods": 0.54,
    "gaming": 0.31,
    "wire_transfer": 0.72,
    "dining": 0.05,
    "utilities": 0.02,
}

ONLINE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "user_avg_spend_24h",
    "user_txn_count_1h",
    "merchant_risk_score",
)

MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "amount",
    "user_avg_spend_24h",
    "user_txn_count_1h",
    "merchant_risk_score",
    "spend_to_avg_ratio",
    "geo_lat",
    "geo_long",
    "geo_dist_from_center_km",
    "hour_sin",
    "hour_cos",
    "merchant_cat_code",
    "device_trust_score",
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Kafka / Redpanda Streaming
    kafka_bootstrap_servers: str = Field(default="localhost:19092")
    kafka_topic_transactions: str = Field(default="financial.transactions.v1")
    kafka_topic_predictions: str = Field(default="financial.fraud_predictions.v1")
    producer_throughput_eps: int = Field(default=500, ge=1)
    fraud_base_rate: float = Field(default=0.015, ge=0.001, le=0.5)

    # Feast Feature Store & Redis Online Store
    feast_repo_path: Path = Field(default=PROJECT_ROOT / "feature_repo")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    use_fakeredis_fallback: bool = Field(default=True)

    # Offline Data & Model Paths
    offline_data_path: Path = Field(default=PROJECT_ROOT / "data" / "offline_transactions.parquet")
    user_features_path: Path = Field(default=PROJECT_ROOT / "data" / "user_features.parquet")
    merchant_features_path: Path = Field(
        default=PROJECT_ROOT / "data" / "merchant_features.parquet"
    )
    reference_baseline_path: Path = Field(
        default=PROJECT_ROOT / "data" / "reference_baseline.parquet"
    )
    duckdb_path: Path = Field(default=PROJECT_ROOT / "data" / "feature_offline.duckdb")
    lightgbm_model_path: Path = Field(default=PROJECT_ROOT / "models" / "fraud_lgbm.txt")
    onnx_model_path: Path = Field(default=PROJECT_ROOT / "models" / "fraud_model.onnx")
    model_metadata_path: Path = Field(default=PROJECT_ROOT / "models" / "model_metadata.json")

    # FastAPI ONNX Serving & SLA
    serving_host: str = Field(default="0.0.0.0")
    serving_port: int = Field(default=8000)
    fraud_probability_threshold: float = Field(default=0.50, ge=0.01, le=0.99)
    fraud_review_threshold: float = Field(default=0.25, ge=0.01, le=0.99)
    sla_p99_latency_ms: float = Field(default=15.0, gt=0.0)
    ort_intra_op_threads: int = Field(default=1, ge=1)
    ort_inter_op_threads: int = Field(default=1, ge=1)

    # Evidently AI & Statistical Drift Monitoring
    drift_window_size: int = Field(default=500, ge=20)
    drift_ks_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    drift_wasserstein_threshold: float = Field(default=0.15, gt=0.0)
    null_spike_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    alert_webhook_url: str = Field(default="http://localhost:8000/v1/drift/webhook-sink")
    alert_webhook_timeout_sec: float = Field(default=3.0, gt=0.0)
    prometheus_port: int = Field(default=9090)

    def ensure_directories(self) -> None:
        """Create required data and model artifact directories if they do not exist."""
        self.offline_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
        (self.feast_repo_path / "data").mkdir(parents=True, exist_ok=True)


settings = Settings()
