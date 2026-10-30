"""Unit tests for Feast feature views, Redis online feature lookups, and DuckDB point-in-time joins."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import fakeredis
import numpy as np
import pandas as pd
from feature_repo.entities import merchant_entity, user_entity
from feature_repo.features import (
    OnlineFeatureStore,
    fraud_feature_service,
    merchant_risk_features_view,
    user_rolling_features_view,
)
from src.training.dataset import build_point_in_time_training_dataset


def test_feast_definitions_and_schema() -> None:
    """Verify Feast entities, feature views, and feature service metadata match specification."""
    assert user_entity.name == "user_id"
    assert merchant_entity.name == "merchant_category"

    user_field_names = {f.name for f in user_rolling_features_view.schema}
    assert {"user_avg_spend_24h", "user_txn_count_1h"}.issubset(user_field_names)

    merchant_field_names = {f.name for f in merchant_risk_features_view.schema}
    assert "merchant_risk_score" in merchant_field_names

    assert fraud_feature_service.name == "fraud_scoring_service_v1"


def test_online_feature_lookup_sub_millisecond() -> None:
    """Verify Redis online feature retrieval returns expected features with sub-5ms latency."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    store = OnlineFeatureStore(redis_client=fake_redis)

    store.upsert_user_features(
        user_id="usr_00101",
        user_avg_spend_24h=142.75,
        user_txn_count_1h=4,
    )
    store.upsert_merchant_risk(merchant_category="crypto_exchange", risk_score=0.82)

    result = store.get_online_features(
        user_id="usr_00101",
        merchant_category="crypto_exchange",
    )

    assert np.isclose(result["user_avg_spend_24h"], 142.75, atol=1e-3)
    assert np.isclose(result["user_txn_count_1h"], 4.0, atol=1e-3)
    assert np.isclose(result["merchant_risk_score"], 0.82, atol=1e-3)
    assert result["lookup_latency_ms"] < 5.0


def test_online_feature_store_batch_and_materialization() -> None:
    """Verify DataFrame materialization and pipelined batch retrieval from Redis online store."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    store = OnlineFeatureStore(redis_client=fake_redis)

    now = datetime.now(UTC)
    user_df = pd.DataFrame(
        [
            {
                "user_id": "usr_00001",
                "user_avg_spend_24h": 50.0,
                "user_txn_count_1h": 1,
                "event_timestamp": now - timedelta(hours=2),
            },
            {
                "user_id": "usr_00001",
                "user_avg_spend_24h": 195.5,
                "user_txn_count_1h": 7,
                "event_timestamp": now - timedelta(minutes=5),
            },
            {
                "user_id": "usr_00002",
                "user_avg_spend_24h": 32.0,
                "user_txn_count_1h": 2,
                "event_timestamp": now - timedelta(minutes=10),
            },
        ]
    )

    written = store.materialize_from_dataframes(user_df=user_df)
    assert written == 2

    batch_res = store.get_online_features_batch(
        [
            ("usr_00001", "electronics"),
            ("usr_00002", "grocery"),
            ("usr_unknown", "retail"),
        ]
    )
    assert len(batch_res) == 3
    # Latest snapshot for usr_00001 must win (195.5, 7)
    assert np.isclose(batch_res[0]["user_avg_spend_24h"], 195.5)
    assert np.isclose(batch_res[0]["user_txn_count_1h"], 7.0)
    assert np.isclose(batch_res[1]["user_avg_spend_24h"], 32.0)
    # Unknown user falls back to safe default (65.0)
    assert np.isclose(batch_res[2]["user_avg_spend_24h"], 65.0)


def test_duckdb_point_in_time_join_prevents_future_leakage() -> None:
    """Verify DuckDB ASOF JOIN strictly joins features at or before transaction timestamp."""
    t0 = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)

    user_df = pd.DataFrame(
        [
            {
                "user_id": "usr_pit",
                "user_avg_spend_24h": 80.0,
                "user_txn_count_1h": 2,
                "event_timestamp": t0 - timedelta(hours=1),
                "created_timestamp": t0 - timedelta(hours=1),
            },
            {
                # Future snapshot relative to transaction — MUST NOT leak into training row
                "user_id": "usr_pit",
                "user_avg_spend_24h": 9999.0,
                "user_txn_count_1h": 99,
                "event_timestamp": t0 + timedelta(hours=1),
                "created_timestamp": t0 + timedelta(hours=1),
            },
        ]
    )
    merchant_df = pd.DataFrame(
        [
            {
                "merchant_category": "retail",
                "merchant_risk_score": 0.08,
                "event_timestamp": t0 - timedelta(days=1),
                "created_timestamp": t0 - timedelta(days=1),
            }
        ]
    )
    transactions_df = pd.DataFrame(
        [
            {
                "transaction_id": "txn_pit_1",
                "user_id": "usr_pit",
                "amount": 120.0,
                "merchant_category": "retail",
                "event_timestamp": t0,
                "geo_lat": 39.8,
                "geo_long": -98.5,
                "device_fingerprint": "dev_trusted_usr_pit",
                "is_fraud": 0,
            }
        ]
    )

    joined = build_point_in_time_training_dataset(
        transactions_df=transactions_df,
        user_df=user_df,
        merchant_df=merchant_df,
    )
    assert len(joined) == 1
    assert np.isclose(float(joined.iloc[0]["user_avg_spend_24h"]), 80.0)
    assert int(joined.iloc[0]["user_txn_count_1h"]) == 2
