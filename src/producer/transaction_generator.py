"""High-throughput synthetic financial transaction generator for Apache Kafka / Redpanda."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np

from src.config import MERCHANT_CATEGORIES, settings

logger = logging.getLogger(__name__)

# Normal merchant category probabilities vs fraud-skewed probabilities
_LEGIT_CATEGORY_WEIGHTS = np.array(
    [0.24, 0.28, 0.06, 0.08, 0.01, 0.02, 0.05, 0.01, 0.16, 0.09],
    dtype=np.float64,
)
_LEGIT_CATEGORY_WEIGHTS /= _LEGIT_CATEGORY_WEIGHTS.sum()

_FRAUD_CATEGORY_WEIGHTS = np.array(
    [0.05, 0.02, 0.10, 0.18, 0.28, 0.16, 0.06, 0.13, 0.01, 0.01],
    dtype=np.float64,
)
_FRAUD_CATEGORY_WEIGHTS /= _FRAUD_CATEGORY_WEIGHTS.sum()


@dataclass(slots=True)
class TransactionEventPayload:
    """Canonical schema for real-time financial transaction events."""

    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    timestamp: str
    geo_lat: float
    geo_long: float
    device_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Convert event dataclass to dictionary."""
        return asdict(self)

    def to_json_bytes(self) -> bytes:
        """Serialize transaction event to UTF-8 JSON bytes for Kafka transmission."""
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")


class SyntheticTransactionGenerator:
    """High-throughput generator of realistic legitimate and fraudulent transactions."""

    def __init__(
        self,
        num_users: int = 2000,
        fraud_rate: float | None = None,
        seed: int = 42,
        inject_drift: bool = False,
    ) -> None:
        self.num_users = num_users
        self.fraud_rate = fraud_rate if fraud_rate is not None else settings.fraud_base_rate
        self.inject_drift = inject_drift
        self.rng = np.random.default_rng(seed)
        self._user_ids = [f"usr_{i:05d}" for i in range(1, num_users + 1)]
        # Pre-compute primary device fingerprints per user
        self._primary_devices = {
            uid: "dev_" + hashlib.sha256(f"trusted:{uid}".encode()).hexdigest()[:16]
            for uid in self._user_ids
        }

    def generate_event(
        self,
        force_fraud: bool | None = None,
        inject_drift: bool | None = None,
    ) -> tuple[TransactionEventPayload, int]:
        """Generate a single synthetic transaction event and its ground-truth fraud label (0 or 1)."""
        drift_active = self.inject_drift if inject_drift is None else inject_drift
        effective_fraud_rate = min(0.35, self.fraud_rate * 6.0) if drift_active else self.fraud_rate

        is_fraud = (
            bool(force_fraud)
            if force_fraud is not None
            else bool(self.rng.random() < effective_fraud_rate)
        )

        user_idx = int(self.rng.integers(0, self.num_users))
        user_id = self._user_ids[user_idx]

        if is_fraud:
            cat_idx = int(self.rng.choice(len(MERCHANT_CATEGORIES), p=_FRAUD_CATEGORY_WEIGHTS))
            merchant_category = MERCHANT_CATEGORIES[cat_idx]
            # High-value account takeover or card-testing burst
            amount = float(np.round(self.rng.lognormal(mean=6.8, sigma=0.75), 2))
            if drift_active:
                amount *= 2.4
            # Geo anomaly: remote coordinates far from US centroid (39.8, -98.6)
            geo_lat = float(np.round(self.rng.uniform(-35.0, 62.0), 5))
            geo_long = float(np.round(self.rng.uniform(10.0, 145.0), 5))
            # Untrusted/spoofed device fingerprint
            nonce = int(self.rng.integers(100_000, 999_999))
            device_fingerprint = (
                "dev_untrusted_" + hashlib.md5(f"{user_id}:{nonce}".encode()).hexdigest()[:12]
            )
        else:
            cat_idx = int(self.rng.choice(len(MERCHANT_CATEGORIES), p=_LEGIT_CATEGORY_WEIGHTS))
            merchant_category = MERCHANT_CATEGORIES[cat_idx]
            mean_log = 4.6 if drift_active else 3.85
            amount = float(np.round(self.rng.lognormal(mean=mean_log, sigma=0.65), 2))
            geo_lat = float(np.round(self.rng.normal(loc=39.8283, scale=3.2), 5))
            geo_long = float(np.round(self.rng.normal(loc=-98.5795, scale=5.5), 5))
            device_fingerprint = self._primary_devices[user_id]

        amount = max(1.0, min(amount, 50_000.0))
        geo_lat = max(-90.0, min(90.0, geo_lat))
        geo_long = max(-180.0, min(180.0, geo_long))

        event = TransactionEventPayload(
            transaction_id=f"txn_{uuid.uuid4().hex[:16]}",
            user_id=user_id,
            amount=round(amount, 2),
            merchant_category=merchant_category,
            timestamp=datetime.now(UTC).isoformat(),
            geo_lat=geo_lat,
            geo_long=geo_long,
            device_fingerprint=device_fingerprint,
        )
        return event, int(is_fraud)

    def stream_events(
        self, count: int | None = None
    ) -> Iterator[tuple[TransactionEventPayload, int]]:
        """Yield an infinite or bounded stream of (TransactionEventPayload, is_fraud) tuples."""
        emitted = 0
        while count is None or emitted < count:
            yield self.generate_event()
            emitted += 1


class KafkaTransactionProducer:
    """Streaming producer publishing synthetic transactions to Apache Kafka / Redpanda."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str | None = None,
        generator: SyntheticTransactionGenerator | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_transactions
        self.generator = generator or SyntheticTransactionGenerator()
        self._delivered_count = 0
        self._error_count = 0
        self._producer = self._init_kafka_producer()

    def _init_kafka_producer(self) -> Any | None:
        """Initialize confluent-kafka Producer if broker configuration is requested."""
        try:
            from confluent_kafka import Producer

            conf = {
                "bootstrap.servers": self.bootstrap_servers,
                "client.id": "fraud-txn-generator-v1",
                "linger.ms": 5,
                "batch.num.messages": 1000,
                "compression.type": "lz4",
                "acks": "1",
                "socket.timeout.ms": 1500,
                "message.timeout.ms": 3000,
            }
            return Producer(conf)
        except Exception as exc:
            logger.warning("Kafka producer initialization skipped (%s); using dry-run mode.", exc)
            return None

    def _delivery_callback(self, err: Any, _msg: Any) -> None:
        if err is not None:
            self._error_count += 1
        else:
            self._delivered_count += 1

    def publish_batch(
        self,
        num_events: int = 100,
        target_http_url: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Publish a batch of synthetic transaction events to Kafka and/or HTTP inference service."""
        t0 = time.perf_counter()
        events_emitted = 0
        fraud_emitted = 0
        http_scored = 0

        http_client = httpx.Client(timeout=2.0) if target_http_url else None
        try:
            for event, is_fraud in self.generator.stream_events(count=num_events):
                events_emitted += 1
                fraud_emitted += is_fraud

                if not dry_run and self._producer is not None:
                    try:
                        self._producer.produce(
                            self.topic,
                            key=event.user_id.encode("utf-8"),
                            value=event.to_json_bytes(),
                            on_delivery=self._delivery_callback,
                        )
                        self._producer.poll(0)
                    except BufferError:
                        self._producer.poll(0.1)

                if http_client is not None and target_http_url:
                    try:
                        resp = http_client.post(target_http_url, json=event.to_dict())
                        if resp.status_code == 200:
                            http_scored += 1
                    except Exception as exc:
                        logger.debug("HTTP scoring call failed: %s", exc)

            if not dry_run and self._producer is not None:
                self._producer.flush(timeout=1.5)
        finally:
            if http_client is not None:
                http_client.close()

        elapsed_sec = max(1e-6, time.perf_counter() - t0)
        return {
            "events_emitted": events_emitted,
            "fraud_events": fraud_emitted,
            "http_scored": http_scored,
            "elapsed_sec": round(elapsed_sec, 4),
            "throughput_eps": round(events_emitted / elapsed_sec, 1),
            "topic": self.topic,
            "inject_drift": self.generator.inject_drift,
        }


def main() -> None:
    """CLI entrypoint for streaming synthetic transactions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Stream synthetic financial transactions to Kafka/Redpanda or FastAPI."
    )
    parser.add_argument("--count", type=int, default=500, help="Number of events to emit.")
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Inject covariate shift and elevated fraud patterns to trigger drift alerts.",
    )
    parser.add_argument(
        "--http-url",
        type=str,
        default="",
        help="Optional FastAPI /v1/predict endpoint URL to score streamed events directly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate events in-memory without pushing to a live Kafka broker.",
    )
    args = parser.parse_args()

    gen = SyntheticTransactionGenerator(inject_drift=args.inject_drift)
    producer = KafkaTransactionProducer(generator=gen)
    stats = producer.publish_batch(
        num_events=args.count,
        target_http_url=args.http_url or None,
        dry_run=args.dry_run,
    )
    logger.info("Streaming simulation completed: %s", json.dumps(stats))


if __name__ == "__main__":
    main()
