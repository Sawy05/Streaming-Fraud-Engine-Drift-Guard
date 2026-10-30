"""Feast feature views, offline sources, and sub-millisecond Redis online lookup client."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import fakeredis
import pandas as pd
import redis
from feast import FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from src.config import MERCHANT_DEFAULT_RISK_SCORES, settings

from feature_repo.entities import merchant_entity, user_entity

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parent
_DATA_DIR = _REPO_DIR.parent / "data"

user_features_source = FileSource(
    name="user_rolling_features_source",
    path=str(_DATA_DIR / "user_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Offline Parquet store for user 24h rolling spend and 1h transaction velocity.",
)

merchant_features_source = FileSource(
    name="merchant_risk_features_source",
    path=str(_DATA_DIR / "merchant_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Offline Parquet store for merchant category baseline chargeback risk scores.",
)

user_rolling_features_view = FeatureView(
    name="user_rolling_features",
    entities=[user_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="user_avg_spend_24h", dtype=Float32),
        Field(name="user_txn_count_1h", dtype=Int64),
    ],
    online=True,
    source=user_features_source,
    tags={"team": "fraud-mlops", "tier": "tier-1-critical"},
)

merchant_risk_features_view = FeatureView(
    name="merchant_risk_features",
    entities=[merchant_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="merchant_risk_score", dtype=Float32),
    ],
    online=True,
    source=merchant_features_source,
    tags={"team": "fraud-mlops", "tier": "tier-1-critical"},
)

fraud_feature_service = FeatureService(
    name="fraud_scoring_service_v1",
    features=[
        user_rolling_features_view,
        merchant_risk_features_view,
    ],
    description="Low-latency online feature bundle for real-time ONNX fraud scoring.",
)


class OnlineFeatureStore:
    """Sub-millisecond Redis online feature store client with Feast schema parity.

    Uses pipelined Redis HGET/HMGET operations to fetch ``user_avg_spend_24h``,
    ``user_txn_count_1h``, and ``merchant_risk_score`` in a single network roundtrip.
    Automatically falls back to an in-memory ``fakeredis`` instance if a live Redis
    broker is unavailable (e.g. during unit tests or standalone CI runs).
    """

    USER_KEY_PREFIX = "feast:online:user:"
    MERCHANT_KEY_PREFIX = "feast:online:merchant:"

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        use_fallback: bool | None = None,
    ) -> None:
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.db = db if db is not None else settings.redis_db
        self.use_fallback = (
            use_fallback if use_fallback is not None else settings.use_fakeredis_fallback
        )
        self.using_fallback = False
        self._client: redis.Redis = redis_client or self._connect()
        self._seed_default_merchant_scores()

    def _connect(self) -> redis.Redis:
        """Connect to live Redis or initialize an in-memory FakeRedis fallback."""
        try:
            client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
            client.ping()
            logger.info("Connected to Redis online store at %s:%s", self.host, self.port)
            return client
        except Exception as exc:
            if not self.use_fallback:
                raise ConnectionError(
                    f"Unable to connect to Redis at {self.host}:{self.port}"
                ) from exc
            logger.info(
                "Live Redis unreachable (%s); activating sub-ms FakeRedis online store.",
                exc,
            )
            self.using_fallback = True
            return fakeredis.FakeRedis(decode_responses=True)

    def _seed_default_merchant_scores(self) -> None:
        """Seed baseline merchant category risk scores into the online store."""
        pipe = self._client.pipeline(transaction=False)
        for category, score in MERCHANT_DEFAULT_RISK_SCORES.items():
            key = f"{self.MERCHANT_KEY_PREFIX}{category}"
            pipe.hsetnx(key, "merchant_risk_score", str(score))
        pipe.execute()

    def materialize_from_dataframes(
        self,
        user_df: pd.DataFrame,
        merchant_df: pd.DataFrame | None = None,
    ) -> int:
        """Materialize latest entity feature values into the Redis online store."""
        records_written = 0
        pipe = self._client.pipeline(transaction=False)

        if not user_df.empty:
            latest_users = user_df.sort_values("event_timestamp").groupby("user_id").last()
            for user_id, row in latest_users.iterrows():
                key = f"{self.USER_KEY_PREFIX}{user_id}"
                pipe.hset(
                    key,
                    mapping={
                        "user_avg_spend_24h": f"{float(row['user_avg_spend_24h']):.4f}",
                        "user_txn_count_1h": str(int(row["user_txn_count_1h"])),
                    },
                )
                records_written += 1

        if merchant_df is not None and not merchant_df.empty:
            latest_merchants = (
                merchant_df.sort_values("event_timestamp").groupby("merchant_category").last()
            )
            for category, row in latest_merchants.iterrows():
                key = f"{self.MERCHANT_KEY_PREFIX}{category}"
                pipe.hset(
                    key,
                    mapping={
                        "merchant_risk_score": f"{float(row['merchant_risk_score']):.4f}",
                    },
                )
                records_written += 1

        pipe.execute()
        return records_written

    def get_online_features(
        self,
        user_id: str,
        merchant_category: str,
    ) -> dict[str, float]:
        """Retrieve online features for a single transaction in sub-millisecond time.

        Returns:
            Dictionary containing ``user_avg_spend_24h``, ``user_txn_count_1h``,
            ``merchant_risk_score``, and ``lookup_latency_ms``.
        """
        t0 = time.perf_counter()
        user_key = f"{self.USER_KEY_PREFIX}{user_id}"
        merchant_key = f"{self.MERCHANT_KEY_PREFIX}{merchant_category}"

        pipe = self._client.pipeline(transaction=False)
        pipe.hmget(user_key, ["user_avg_spend_24h", "user_txn_count_1h"])
        pipe.hget(merchant_key, "merchant_risk_score")
        user_vals, merchant_val = pipe.execute()

        avg_spend = float(user_vals[0]) if user_vals and user_vals[0] is not None else 65.0
        txn_count = float(user_vals[1]) if user_vals and user_vals[1] is not None else 1.0
        merchant_risk = (
            float(merchant_val)
            if merchant_val is not None
            else MERCHANT_DEFAULT_RISK_SCORES.get(merchant_category, 0.15)
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "user_avg_spend_24h": avg_spend,
            "user_txn_count_1h": txn_count,
            "merchant_risk_score": merchant_risk,
            "lookup_latency_ms": latency_ms,
        }

    def get_online_features_batch(
        self,
        entities: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        """Retrieve online features for a batch of (user_id, merchant_category) pairs."""
        if not entities:
            return []

        pipe = self._client.pipeline(transaction=False)
        for user_id, merchant_category in entities:
            pipe.hmget(
                f"{self.USER_KEY_PREFIX}{user_id}",
                ["user_avg_spend_24h", "user_txn_count_1h"],
            )
            pipe.hget(
                f"{self.MERCHANT_KEY_PREFIX}{merchant_category}",
                "merchant_risk_score",
            )
        raw_results = pipe.execute()

        features_list: list[dict[str, float]] = []
        for idx, (_user_id, merchant_category) in enumerate(entities):
            user_vals = raw_results[2 * idx]
            merchant_val = raw_results[2 * idx + 1]
            avg_spend = float(user_vals[0]) if user_vals and user_vals[0] is not None else 65.0
            txn_count = float(user_vals[1]) if user_vals and user_vals[1] is not None else 1.0
            merchant_risk = (
                float(merchant_val)
                if merchant_val is not None
                else MERCHANT_DEFAULT_RISK_SCORES.get(merchant_category, 0.15)
            )
            features_list.append(
                {
                    "user_avg_spend_24h": avg_spend,
                    "user_txn_count_1h": txn_count,
                    "merchant_risk_score": merchant_risk,
                }
            )
        return features_list

    def upsert_user_features(
        self,
        user_id: str,
        user_avg_spend_24h: float,
        user_txn_count_1h: int,
    ) -> None:
        """Update a single user's rolling features in the Redis online store."""
        key = f"{self.USER_KEY_PREFIX}{user_id}"
        self._client.hset(
            key,
            mapping={
                "user_avg_spend_24h": f"{float(user_avg_spend_24h):.4f}",
                "user_txn_count_1h": str(int(user_txn_count_1h)),
            },
        )

    def upsert_merchant_risk(self, merchant_category: str, risk_score: float) -> None:
        """Update a merchant category's risk score in the Redis online store."""
        key = f"{self.MERCHANT_KEY_PREFIX}{merchant_category}"
        self._client.hset(key, "merchant_risk_score", f"{float(risk_score):.4f}")

    def health_check(self) -> dict[str, Any]:
        """Return connectivity status of the Redis online feature store."""
        try:
            pong = bool(self._client.ping())
        except Exception:
            pong = False
        return {
            "healthy": pong,
            "backend": "fakeredis" if self.using_fallback else "redis",
            "host": self.host,
            "port": self.port,
        }
