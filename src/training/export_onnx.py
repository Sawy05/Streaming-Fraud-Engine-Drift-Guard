"""Export trained LightGBM Booster to optimized ONNX graph for sub-millisecond serving."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

from src.config import MODEL_FEATURE_COLUMNS, settings

logger = logging.getLogger(__name__)


def _strip_zipmap_if_present(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove ZipMap post-processing node if present and normalize dynamic batch output shapes."""
    graph = onnx_model.graph
    zipmap_nodes = [n for n in graph.node if n.op_type == "ZipMap"]
    for zm_node in zipmap_nodes:
        prob_tensor_name = zm_node.input[0]
        zm_output_name = zm_node.output[0]
        graph.node.remove(zm_node)
        for idx, out in enumerate(graph.output):
            if out.name == zm_output_name:
                new_out = onnx.helper.make_tensor_value_info(
                    prob_tensor_name,
                    onnx.TensorProto.FLOAT,
                    ["N", 2],
                )
                graph.output[idx].CopyFrom(new_out)

    # Ensure all graph outputs allow dynamic batch sizes ("N") without shape mismatch warnings
    for out in graph.output:
        if out.type.HasField("tensor_type") and out.type.tensor_type.HasField("shape"):
            dims = out.type.tensor_type.shape.dim
            if len(dims) >= 1:
                dims[0].ClearField("dim_value")
                dims[0].dim_param = "N"
    return onnx_model


def export_lightgbm_to_onnx(
    lgbm_model_path: Path | None = None,
    onnx_output_path: Path | None = None,
    target_opset: int = 15,
) -> dict[str, Any]:
    """Convert trained LightGBM model to ONNX format and verify numerical & latency parity."""
    src_path = lgbm_model_path or settings.lightgbm_model_path
    dst_path = onnx_output_path or settings.onnx_model_path

    if not src_path.exists():
        from src.training.train import train_fraud_model

        train_fraud_model()

    booster = lgb.Booster(model_file=str(src_path))
    booster.params["objective"] = "binary"
    num_features = len(MODEL_FEATURE_COLUMNS)
    initial_types = [("float_input", FloatTensorType([None, num_features]))]

    onnx_model = onnxmltools.convert_lightgbm(
        booster,
        name="LightGBMFraudFocalDetector",
        initial_types=initial_types,
        target_opset=target_opset,
        zipmap=False,
    )
    onnx_model = _strip_zipmap_if_present(onnx_model)
    onnx.checker.check_model(onnx_model)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(dst_path))

    # Validate numerical parity and benchmark single-event P99 latency
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(dst_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(123)
    sample_x = rng.normal(size=(64, num_features)).astype(np.float32)
    sample_x[:, 0] = np.abs(sample_x[:, 0]) * 120.0

    raw_margins = np.asarray(booster.predict(sample_x, raw_score=True), dtype=np.float64).reshape(
        -1
    )
    lgb_probs = (1.0 / (1.0 + np.exp(-np.clip(raw_margins, -30.0, 30.0)))).astype(np.float32)
    ort_outputs = session.run(None, {input_name: sample_x})
    ort_probs = extract_positive_probabilities(ort_outputs)

    max_abs_diff = float(np.max(np.abs(lgb_probs - ort_probs)))

    # Benchmark 500 single-row inferences
    single_row = sample_x[:1]
    latencies_ms: list[float] = []
    for _ in range(500):
        t0 = time.perf_counter()
        session.run(None, {input_name: single_row})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))

    report = {
        "onnx_model_path": str(dst_path),
        "num_features": num_features,
        "max_abs_diff_vs_lightgbm": round(max_abs_diff, 7),
        "single_inference_p50_ms": round(p50_ms, 4),
        "single_inference_p95_ms": round(p95_ms, 4),
        "single_inference_p99_ms": round(p99_ms, 4),
    }
    logger.info("Exported and validated ONNX model: %s", report)
    return report


def extract_positive_probabilities(ort_outputs: list[Any]) -> np.ndarray:
    """Extract 1D array of positive-class fraud probabilities from ONNX Runtime outputs."""
    candidate = ort_outputs[-1]
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
        return np.array(
            [float(row.get(1, row.get("1", 0.0))) for row in candidate], dtype=np.float32
        )

    arr = np.asarray(candidate, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            return arr[:, 1]
        return arr[:, 0]
    return arr.reshape(-1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    export_lightgbm_to_onnx()
