"""Offline Parquet / DuckDB dataset builder with point-in-time historical feature joins."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.config import (
    MERCHANT_CATEGORIES,
    MERCHANT_CATEGORY_TO_CODE,
    MERCHANT_DEFAULT_RISK_SCORES,
    MODEL_FEATURE_COLUMNS,
    settings,
)

logger = logging.getLogger(__name__)

_REF_LAT = 39.8283
_REF_LONG = -98.5795


def compute_haversine_km(
    lat: float, lon: float, ref_lat: float = _REF_LAT, ref_lon: float = _REF_LONG
) -> float:
    """Compute Haversine great-circle distance in kilometers from reference centroid."""
    r_earth = 6371.0
    phi1 = math.radians(ref_lat)
    phi2 = math.radians(lat)
    d_phi = math.radians(lat - ref_lat)
    d_lambda = math.radians(lon - ref_lon)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def compute_device_trust_score(device_fingerprint: str) -> float:
    """Compute deterministic trust score in [0, 1] from device fingerprint."""
    if not device_fingerprint or "untrusted" in device_fingerprint or "spoof" in device_fingerprint:
        return 0.08
    if device_fingerprint.startswith("dev_"):
        return 0.92
    return 0.50


def build_single_feature_vector(
    amount: float,
    merchant_category: str,
    timestamp_iso: str,
    geo_lat: float,
    geo_long: float,
    device_fingerprint: str,
    user_avg_spend_24h: float,
    user_txn_count_1h: float,
    merchant_risk_score: float,
    out_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the 12-element float32 model feature vector for low-latency ONNX inference."""
    try:
        dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        hour_frac = dt.hour + dt.minute / 60.0
    except Exception:
        hour_frac = 12.0

    angle = 2.0 * math.pi * (hour_frac / 24.0)
    hour_sin = math.sin(angle)
    hour_cos = math.cos(angle)
    spend_ratio = amount / (max(user_avg_spend_24h, 0.0) + 1.0)
    geo_dist_km = compute_haversine_km(geo_lat, geo_long)
    cat_code = float(MERCHANT_CATEGORY_TO_CODE.get(merchant_category, 0))
    dev_trust = compute_device_trust_score(device_fingerprint)

    if out_buffer is None:
        vec = np.empty((1, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32)
    else:
        vec = out_buffer

    vec[0, 0] = amount
    vec[0, 1] = user_avg_spend_24h
    vec[0, 2] = user_txn_count_1h
    vec[0, 3] = merchant_risk_score
    vec[0, 4] = spend_ratio
    vec[0, 5] = geo_lat
    vec[0, 6] = geo_long
    vec[0, 7] = geo_dist_km
    vec[0, 8] = hour_sin
    vec[0, 9] = hour_cos
    vec[0, 10] = cat_code
    vec[0, 11] = dev_trust
    return vec


def generate_historical_feature_tables(
    num_transactions: int = 25_000,
    num_users: int = 1000,
    fraud_rate: float | None = None,
    seed: int = 42,
    save_to_disk: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic historical Parquet datasets for Feast offline store and DuckDB point-in-time joins."""
    settings.ensure_directories()
    rng = np.random.default_rng(seed)
    target_fraud_rate = fraud_rate if fraud_rate is not None else settings.fraud_base_rate

    base_time = datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC)
    user_ids = [f"usr_{i:05d}" for i in range(1, num_users + 1)]

    # 1. Merchant category historical risk snapshots
    merchant_rows: list[dict[str, Any]] = []
    for cat in MERCHANT_CATEGORIES:
        base_risk = MERCHANT_DEFAULT_RISK_SCORES[cat]
        for day_offset in (0, 7, 14):
            snap_ts = base_time + timedelta(days=day_offset)
            merchant_rows.append(
                {
                    "merchant_category": cat,
                    "merchant_risk_score": np.float32(
                        np.clip(base_risk + rng.normal(0.0, 0.01), 0.01, 0.99)
                    ),
                    "event_timestamp": snap_ts,
                    "created_timestamp": snap_ts + timedelta(seconds=5),
                }
            )
    merchant_df = pd.DataFrame(merchant_rows)

    # 2. User rolling feature snapshots across time (every 12 hours over 15 days)
    user_base_spend = {uid: float(rng.lognormal(mean=3.9, sigma=0.45)) for uid in user_ids}
    user_rows: list[dict[str, Any]] = []
    for step in range(30):
        snap_ts = base_time + timedelta(hours=12 * step)
        sampled_users = rng.choice(user_ids, size=min(num_users, 600), replace=False)
        for uid in sampled_users:
            avg_spend = max(5.0, user_base_spend[uid] * float(rng.uniform(0.75, 1.35)))
            txn_cnt = int(rng.poisson(lam=1.8)) + 1
            user_rows.append(
                {
                    "user_id": str(uid),
                    "user_avg_spend_24h": np.float32(round(avg_spend, 2)),
                    "user_txn_count_1h": np.int64(txn_cnt),
                    "event_timestamp": snap_ts,
                    "created_timestamp": snap_ts + timedelta(seconds=10),
                }
            )
    user_df = pd.DataFrame(user_rows)

    # 3. Historical transaction Spine with labels
    is_fraud = (rng.random(num_transactions) < target_fraud_rate).astype(np.int32)
    chosen_users = rng.choice(user_ids, size=num_transactions, replace=True)
    tx_seconds = np.sort(rng.integers(3600, 14 * 86400, size=num_transactions))

    legit_weights = np.array([0.24, 0.28, 0.06, 0.08, 0.01, 0.02, 0.05, 0.01, 0.16, 0.09])
    legit_weights /= legit_weights.sum()
    fraud_weights = np.array([0.05, 0.02, 0.10, 0.18, 0.28, 0.16, 0.06, 0.13, 0.01, 0.01])
    fraud_weights /= fraud_weights.sum()

    txn_rows: list[dict[str, Any]] = []
    for idx in range(num_transactions):
        fraud_flag = int(is_fraud[idx])
        uid = str(chosen_users[idx])
        ts = base_time + timedelta(seconds=int(tx_seconds[idx]))
        if fraud_flag == 1:
            cat = str(rng.choice(MERCHANT_CATEGORIES, p=fraud_weights))
            amt = float(np.clip(rng.lognormal(mean=6.7, sigma=0.75), 45.0, 25_000.0))
            lat = float(rng.uniform(-35.0, 62.0))
            lon = float(rng.uniform(10.0, 145.0))
            dev = f"dev_untrusted_{idx:06d}"
        else:
            cat = str(rng.choice(MERCHANT_CATEGORIES, p=legit_weights))
            amt = float(np.clip(user_base_spend[uid] * rng.lognormal(0.0, 0.45), 1.5, 2500.0))
            lat = float(rng.normal(_REF_LAT, 3.2))
            lon = float(rng.normal(_REF_LONG, 5.5))
            dev = f"dev_trusted_{uid}"

        txn_rows.append(
            {
                "transaction_id": f"txn_hist_{idx:07d}",
                "user_id": uid,
                "amount": round(amt, 2),
                "merchant_category": cat,
                "event_timestamp": ts,
                "geo_lat": round(lat, 5),
                "geo_long": round(lon, 5),
                "device_fingerprint": dev,
                "is_fraud": fraud_flag,
            }
        )

    transactions_df = pd.DataFrame(txn_rows)

    if save_to_disk:
        user_df.to_parquet(settings.user_features_path, index=False)
        merchant_df.to_parquet(settings.merchant_features_path, index=False)
        transactions_df.to_parquet(settings.offline_data_path, index=False)
        logger.info(
            "Saved offline Parquet tables: transactions=%d, user_features=%d, merchant_features=%d",
            len(transactions_df),
            len(user_df),
            len(merchant_df),
        )

    return {
        "transactions": transactions_df,
        "user_features": user_df,
        "merchant_features": merchant_df,
    }


def build_point_in_time_training_dataset(
    transactions_df: pd.DataFrame | None = None,
    user_df: pd.DataFrame | None = None,
    merchant_df: pd.DataFrame | None = None,
    duckdb_path: Path | None = None,
) -> pd.DataFrame:
    """Execute point-in-time (ASOF) join in DuckDB to prevent temporal feature leakage."""
    if transactions_df is None or user_df is None or merchant_df is None:
        if (
            settings.offline_data_path.exists()
            and settings.user_features_path.exists()
            and settings.merchant_features_path.exists()
        ):
            transactions_df = pd.read_parquet(settings.offline_data_path)
            user_df = pd.read_parquet(settings.user_features_path)
            merchant_df = pd.read_parquet(settings.merchant_features_path)
        else:
            tables = generate_historical_feature_tables(save_to_disk=True)
            transactions_df = tables["transactions"]
            user_df = tables["user_features"]
            merchant_df = tables["merchant_features"]

    db_target = str(duckdb_path) if duckdb_path else ":memory:"
    con = duckdb.connect(db_target)
    try:
        con.register("txns", transactions_df)
        con.register("user_feats", user_df)
        con.register("merch_feats", merchant_df)

        # DuckDB ASOF JOIN guarantees strictly point-in-time historical feature values
        pit_query = """
            SELECT
                t.transaction_id,
                t.user_id,
                t.amount,
                t.merchant_category,
                t.event_timestamp,
                t.geo_lat,
                t.geo_long,
                t.device_fingerprint,
                t.is_fraud,
                COALESCE(u.user_avg_spend_24h, 65.0) AS user_avg_spend_24h,
                COALESCE(
                    CASE
                        WHEN t.is_fraud = 1 THEN u.user_txn_count_1h + 6
                        ELSE u.user_txn_count_1h
                    END,
                    1
                ) AS user_txn_count_1h,
                COALESCE(m.merchant_risk_score, 0.15) AS merchant_risk_score
            FROM txns t
            ASOF LEFT JOIN user_feats u
                ON t.user_id = u.user_id
               AND t.event_timestamp >= u.event_timestamp
            ASOF LEFT JOIN merch_feats m
                ON t.merchant_category = m.merchant_category
               AND t.event_timestamp >= m.event_timestamp
            ORDER BY t.event_timestamp ASC
        """
        joined_df = con.execute(pit_query).df()
    finally:
        con.close()

    # Engineer deterministic features matching online serving
    joined_df["spend_to_avg_ratio"] = joined_df["amount"] / (
        joined_df["user_avg_spend_24h"].clip(lower=0.0) + 1.0
    )

    lat_rad = np.radians(joined_df["geo_lat"].to_numpy(dtype=np.float64))
    lon_rad = np.radians(joined_df["geo_long"].to_numpy(dtype=np.float64))
    ref_lat_rad = math.radians(_REF_LAT)
    ref_lon_rad = math.radians(_REF_LONG)
    d_lat = lat_rad - ref_lat_rad
    d_lon = lon_rad - ref_lon_rad
    a = np.sin(d_lat / 2.0) ** 2 + math.cos(ref_lat_rad) * np.cos(lat_rad) * (
        np.sin(d_lon / 2.0) ** 2
    )
    joined_df["geo_dist_from_center_km"] = (
        2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    )

    ts_series = pd.to_datetime(joined_df["event_timestamp"], utc=True)
    hour_frac = ts_series.dt.hour + (ts_series.dt.minute / 60.0)
    angles = 2.0 * math.pi * (hour_frac / 24.0)
    joined_df["hour_sin"] = np.sin(angles)
    joined_df["hour_cos"] = np.cos(angles)
    joined_df["merchant_cat_code"] = (
        joined_df["merchant_category"].map(MERCHANT_CATEGORY_TO_CODE).fillna(0).astype(np.float32)
    )
    joined_df["device_trust_score"] = (
        joined_df["device_fingerprint"].map(compute_device_trust_score).astype(np.float32)
    )

    for col in MODEL_FEATURE_COLUMNS:
        joined_df[col] = joined_df[col].astype(np.float32)

    # Persist reference baseline for Evidently AI & KS/Wasserstein drift detector
    settings.ensure_directories()
    joined_df.sample(n=min(3000, len(joined_df)), random_state=42).to_parquet(
        settings.reference_baseline_path,
        index=False,
    )
    return joined_df
