"""Sub-millisecond ONNX Runtime fraud inference engine integrated with Feast / Redis online features."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from feature_repo.features import OnlineFeatureStore

from src.config import MODEL_FEATURE_COLUMNS, settings
from src.serving.schemas import (
    BatchFraudPredictionResponse,
    FraudPredictionResponse,
    OnlineFeaturesSnapshot,
    TransactionRequest,
)
from src.training.dataset import build_single_feature_vector
from src.training.export_onnx import (
    export_lightgbm_to_onnx,
    extract_positive_probabilities,
)

logger = logging.getLogger(__name__)


class ONNXFraudEngine:
    """High-throughput, sub-15ms P99 ONNX Runtime inference engine for transaction fraud scoring."""

    def __init__(
        self,
        onnx_model_path: Path | None = None,
        online_store: OnlineFeatureStore | None = None,
        fraud_threshold: float | None = None,
        review_threshold: float | None = None,
    ) -> None:
        self.model_path = onnx_model_path or settings.onnx_model_path
        self.online_store = online_store or OnlineFeatureStore()
        self.fraud_threshold = (
            fraud_threshold if fraud_threshold is not None else settings.fraud_probability_threshold
        )
        self.review_threshold = (
            review_threshold if review_threshold is not None else settings.fraud_review_threshold
        )
        self._num_features = len(MODEL_FEATURE_COLUMNS)
        self._session = self._load_and_warmup_session()
        self._input_name = self._session.get_inputs()[0].name

    def _load_and_warmup_session(self) -> ort.InferenceSession:
        """Load ONNX model (exporting first if missing) and warm up the CPU execution provider."""
        if not self.model_path.exists():
            logger.info("ONNX model not found at %s; training & exporting now...", self.model_path)
            export_lightgbm_to_onnx(onnx_output_path=self.model_path)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = settings.ort_intra_op_threads
        sess_options.inter_op_num_threads = settings.ort_inter_op_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        dummy = np.zeros((1, self._num_features), dtype=np.float32)
        for _ in range(10):
            session.run(None, {input_name: dummy})
        return session

    def score_feature_matrix(self, features_2d: np.ndarray) -> np.ndarray:
        """Run raw ONNX Runtime inference on a 2D float32 feature matrix of shape (N, 12)."""
        x_contig = np.ascontiguousarray(features_2d, dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: x_contig})
        probs = extract_positive_probabilities(outputs)
        return np.clip(probs, 0.0, 1.0)

    def _classify_decision(self, probability: float) -> tuple[bool, str]:
        if probability >= self.fraud_threshold:
            return True, "BLOCK"
        if probability >= self.review_threshold:
            return False, "REVIEW"
        return False, "APPROVE"

    def predict(
        self,
        request: TransactionRequest,
    ) -> tuple[FraudPredictionResponse, dict[str, float]]:
        """Score a single transaction event end-to-end (online feature lookup + ONNX inference).

        Returns:
            Tuple of ``(FraudPredictionResponse, monitoring_payload_dict)``.
        """
        t_start = time.perf_counter()

        online_feats = self.online_store.get_online_features(
            user_id=request.user_id,
            merchant_category=request.merchant_category,
        )

        t_infer_start = time.perf_counter()
        feature_vec = build_single_feature_vector(
            amount=request.amount,
            merchant_category=request.merchant_category,
            timestamp_iso=request.timestamp,
            geo_lat=request.geo_lat,
            geo_long=request.geo_long,
            device_fingerprint=request.device_fingerprint,
            user_avg_spend_24h=online_feats["user_avg_spend_24h"],
            user_txn_count_1h=online_feats["user_txn_count_1h"],
            merchant_risk_score=online_feats["merchant_risk_score"],
        )
        outputs = self._session.run(None, {self._input_name: feature_vec})
        prob = float(np.clip(extract_positive_probabilities(outputs)[0], 0.0, 1.0))
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        is_fraud, decision = self._classify_decision(prob)

        response = FraudPredictionResponse(
            transaction_id=request.transaction_id,
            user_id=request.user_id,
            fraud_probability=round(prob, 6),
            is_fraud=is_fraud,
            decision=decision,  # type: ignore[arg-type]
            threshold_used=self.fraud_threshold,
            inference_latency_ms=round(infer_latency_ms, 4),
            total_latency_ms=round(total_latency_ms, 4),
            online_features=OnlineFeaturesSnapshot(
                user_avg_spend_24h=round(online_feats["user_avg_spend_24h"], 4),
                user_txn_count_1h=round(online_feats["user_txn_count_1h"], 2),
                merchant_risk_score=round(online_feats["merchant_risk_score"], 4),
                lookup_latency_ms=round(online_feats["lookup_latency_ms"], 4),
            ),
        )

        monitoring_payload = {
            "amount": float(feature_vec[0, 0]),
            "user_avg_spend_24h": float(feature_vec[0, 1]),
            "user_txn_count_1h": float(feature_vec[0, 2]),
            "merchant_risk_score": float(feature_vec[0, 3]),
            "spend_to_avg_ratio": float(feature_vec[0, 4]),
            "geo_dist_from_center_km": float(feature_vec[0, 7]),
            "fraud_probability": prob,
        }
        return response, monitoring_payload

    def predict_batch(
        self,
        transactions: list[TransactionRequest],
    ) -> tuple[BatchFraudPredictionResponse, list[dict[str, float]]]:
        """Vectorized batch prediction using pipelined Redis lookups and batch ONNX execution."""
        t0 = time.perf_counter()
        n = len(transactions)
        entities = [(t.user_id, t.merchant_category) for t in transactions]

        t_lookup = time.perf_counter()
        online_batch = self.online_store.get_online_features_batch(entities)
        lookup_per_event_ms = ((time.perf_counter() - t_lookup) * 1000.0) / max(1, n)

        matrix = np.empty((n, self._num_features), dtype=np.float32)
        for idx, (req, ofeats) in enumerate(zip(transactions, online_batch, strict=True)):
            build_single_feature_vector(
                amount=req.amount,
                merchant_category=req.merchant_category,
                timestamp_iso=req.timestamp,
                geo_lat=req.geo_lat,
                geo_long=req.geo_long,
                device_fingerprint=req.device_fingerprint,
                user_avg_spend_24h=ofeats["user_avg_spend_24h"],
                user_txn_count_1h=ofeats["user_txn_count_1h"],
                merchant_risk_score=ofeats["merchant_risk_score"],
                out_buffer=matrix[idx : idx + 1],
            )

        t_infer = time.perf_counter()
        probs = self.score_feature_matrix(matrix)
        infer_per_event_ms = ((time.perf_counter() - t_infer) * 1000.0) / max(1, n)
        total_ms = (time.perf_counter() - t0) * 1000.0
        mean_event_ms = total_ms / max(1, n)

        predictions: list[FraudPredictionResponse] = []
        monitoring_rows: list[dict[str, float]] = []
        for idx, (req, ofeats) in enumerate(zip(transactions, online_batch, strict=True)):
            prob = float(probs[idx])
            is_fraud, decision = self._classify_decision(prob)
            predictions.append(
                FraudPredictionResponse(
                    transaction_id=req.transaction_id,
                    user_id=req.user_id,
                    fraud_probability=round(prob, 6),
                    is_fraud=is_fraud,
                    decision=decision,  # type: ignore[arg-type]
                    threshold_used=self.fraud_threshold,
                    inference_latency_ms=round(infer_per_event_ms, 4),
                    total_latency_ms=round(mean_event_ms, 4),
                    online_features=OnlineFeaturesSnapshot(
                        user_avg_spend_24h=round(ofeats["user_avg_spend_24h"], 4),
                        user_txn_count_1h=round(ofeats["user_txn_count_1h"], 2),
                        merchant_risk_score=round(ofeats["merchant_risk_score"], 4),
                        lookup_latency_ms=round(lookup_per_event_ms, 4),
                    ),
                )
            )
            monitoring_rows.append(
                {
                    "amount": float(matrix[idx, 0]),
                    "user_avg_spend_24h": float(matrix[idx, 1]),
                    "user_txn_count_1h": float(matrix[idx, 2]),
                    "merchant_risk_score": float(matrix[idx, 3]),
                    "spend_to_avg_ratio": float(matrix[idx, 4]),
                    "geo_dist_from_center_km": float(matrix[idx, 7]),
                    "fraud_probability": prob,
                }
            )

        return (
            BatchFraudPredictionResponse(
                predictions=predictions,
                batch_size=n,
                total_latency_ms=round(total_ms, 4),
                mean_latency_per_event_ms=round(mean_event_ms, 4),
            ),
            monitoring_rows,
        )

    @property
    def is_loaded(self) -> bool:
        """Return True if the ONNX Runtime session is active."""
        return self._session is not None
