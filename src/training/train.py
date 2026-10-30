"""LightGBM training pipeline with Focal Loss and PR-AUC optimization for imbalanced fraud data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import MODEL_FEATURE_COLUMNS, settings
from src.training.dataset import build_point_in_time_training_dataset

logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid function."""
    x_clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class FocalLossObjective:
    """Analytical Binary Focal Loss objective for LightGBM addressing extreme class imbalance.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def __call__(
        self,
        y_pred: np.ndarray,
        dataset: lgb.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute first-order gradient and second-order Hessian wrt raw logits."""
        y_true = dataset.get_label().astype(np.float64)
        p = sigmoid(y_pred.astype(np.float64))
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)

        # p_t = p if y == 1 else (1 - p)
        p_t = np.where(y_true == 1.0, p, 1.0 - p)
        alpha_t = np.where(y_true == 1.0, self.alpha, 1.0 - self.alpha)
        y_sign = np.where(y_true == 1.0, 1.0, -1.0)

        # Modulation factor (1 - p_t)^gamma
        mod_factor = np.power(1.0 - p_t, self.gamma)
        log_pt = np.log(p_t)

        # dL/dz
        grad = alpha_t * y_sign * mod_factor * (self.gamma * p_t * log_pt + p_t - 1.0)

        # Positive definite second-order approximation for stable Newton boosting steps
        u = self.gamma * p_t * log_pt + p_t - 1.0
        du_dpt = self.gamma * (log_pt + 1.0) + 1.0
        dpt_dz = p * (1.0 - p)
        hess = (
            alpha_t
            * dpt_dz
            * np.abs(
                -self.gamma * np.power(1.0 - p_t, max(0.0, self.gamma - 1.0)) * u
                + mod_factor * du_dpt
            )
        )
        hess = np.maximum(hess, 1e-5)
        return grad, hess


def pr_auc_eval(
    y_pred: np.ndarray,
    dataset: lgb.Dataset,
) -> tuple[str, float, bool]:
    """Custom LightGBM evaluation metric computing Area Under the Precision-Recall Curve (PR-AUC)."""
    y_true = dataset.get_label()
    probs = sigmoid(y_pred)
    score = float(average_precision_score(y_true, probs))
    return "pr_auc", score, True


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Find decision threshold maximizing F1 score across the Precision-Recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = (2.0 * precisions[:-1] * recalls[:-1]) / np.maximum(
        precisions[:-1] + recalls[:-1], 1e-8
    )
    best_idx = int(np.argmax(f1_scores))
    return float(np.clip(thresholds[best_idx], 0.15, 0.85)), float(f1_scores[best_idx])


def _patch_booster_binary_sigmoid(model_path: Path) -> None:
    """Ensure saved LightGBM model file specifies `objective=binary sigmoid:1`.

    When LightGBM trains with a custom Python objective, the serialized model writes
    `objective=custom` and `[objective: custom]`. Patching both to `binary` enables native
    logistic post-transformation in both `lgb.Booster.predict()` and ONNX `TreeEnsembleClassifier`.
    """
    text = model_path.read_text(encoding="utf-8")
    patched = (
        text.replace("objective=none", "objective=binary sigmoid:1")
        .replace("objective=custom", "objective=binary sigmoid:1")
        .replace("[objective: custom]", "[objective: binary]")
        .replace("[objective: none]", "[objective: binary]")
    )
    model_path.write_text(patched, encoding="utf-8")


def train_fraud_model(
    df: pd.DataFrame | None = None,
    num_boost_round: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the LightGBM fraud detector with Focal Loss & PR-AUC optimization and save artifacts."""
    settings.ensure_directories()
    if df is None:
        df = build_point_in_time_training_dataset()

    feature_cols = list(MODEL_FEATURE_COLUMNS)
    x_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["is_fraud"].to_numpy(dtype=np.int32)

    x_train, x_val, y_train, y_val = train_test_split(
        x_all,
        y_all,
        test_size=0.20,
        random_state=seed,
        stratify=y_all,
    )

    focal_loss = FocalLossObjective(alpha=0.75, gamma=2.0)
    dtrain = lgb.Dataset(x_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, feature_name=feature_cols)

    params: dict[str, Any] = {
        "objective": focal_loss,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.08,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": 2,
        "seed": seed,
    }

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        feval=pr_auc_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    # Save and patch model header for native logistic sigmoid post-transform
    settings.lightgbm_model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(settings.lightgbm_model_path))
    _patch_booster_binary_sigmoid(settings.lightgbm_model_path)

    # Reload patched booster and compute calibrated Focal-Loss probabilities via logistic sigmoid
    calibrated_booster = lgb.Booster(model_file=str(settings.lightgbm_model_path))
    raw_val_preds = np.asarray(calibrated_booster.predict(x_val), dtype=np.float64)
    val_probs = sigmoid(raw_val_preds)

    pr_auc = float(average_precision_score(y_val, val_probs))
    roc_auc = float(roc_auc_score(y_val, val_probs))
    best_thresh, best_f1 = find_optimal_threshold(y_val, val_probs)
    default_f1 = float(f1_score(y_val, (val_probs >= 0.50).astype(int)))

    metadata: dict[str, Any] = {
        "model_type": "LightGBM_FocalLoss_BinaryClassifier",
        "focal_loss_alpha": focal_loss.alpha,
        "focal_loss_gamma": focal_loss.gamma,
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "positive_rate": round(float(y_all.mean()), 5),
        "val_pr_auc": round(pr_auc, 5),
        "val_roc_auc": round(roc_auc, 5),
        "val_best_f1": round(best_f1, 5),
        "val_f1_at_050": round(default_f1, 5),
        "optimal_threshold": round(best_thresh, 4),
        "num_trees": calibrated_booster.num_trees(),
    }

    settings.model_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Trained LightGBM Focal Loss model: %s", json.dumps(metadata))
    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    train_fraud_model()
    from src.training.export_onnx import export_lightgbm_to_onnx

    export_lightgbm_to_onnx()
