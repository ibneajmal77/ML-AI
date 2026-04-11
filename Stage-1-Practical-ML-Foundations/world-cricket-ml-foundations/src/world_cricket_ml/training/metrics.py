from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = (probabilities >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, labels)), 4),
        "precision": round(float(precision_score(y_true, labels, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, labels, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, labels, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4) if len(set(y_true)) > 1 else 0.0,
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
    }


def regression_metrics(y_true, predictions) -> dict:
    """Return MAE, RMSE, R², and MAPE.

    MAPE excludes rows where the actual value is zero to avoid division by
    zero.  Returns None for MAPE when all actuals are zero.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(predictions, dtype=float)
    mse = float(((y_true_arr - pred_arr) ** 2).mean())

    nonzero_mask = y_true_arr != 0.0
    if nonzero_mask.sum() > 0:
        mape: float | None = float(
            np.abs((y_true_arr[nonzero_mask] - pred_arr[nonzero_mask]) / y_true_arr[nonzero_mask]).mean()
        )
    else:
        mape = None

    return {
        "mae": round(float(mean_absolute_error(y_true_arr, pred_arr)), 4),
        "rmse": round(math.sqrt(mse), 4),
        "r2": round(float(r2_score(y_true_arr, pred_arr)), 4),
        "mape": round(mape, 4) if mape is not None else None,
    }


def save_calibration_curve(y_true, raw_probs, calibrated_probs, artifact_path) -> None:
    """Save calibration curve data for raw and calibrated predictions as a CSV.

    Each row is one (model_type, bin) pair with the mean predicted probability
    and the actual fraction of positives in that bin.  A perfectly calibrated
    model has fraction_of_positives ≈ mean_predicted_value on the diagonal.
    """
    try:
        rows = []
        n_bins = max(2, min(8, int(len(y_true) / 5)))
        for model_type, probs in [("raw", raw_probs), ("calibrated", calibrated_probs)]:
            fop, mpv = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
            for f, m in zip(fop, mpv):
                rows.append({
                    "model_type": model_type,
                    "mean_predicted_value": round(float(m), 4),
                    "fraction_of_positives": round(float(f), 4),
                })
        if rows:
            pd.DataFrame(rows).to_csv(artifact_path, index=False)
            log.info("Calibration curve saved: %d data points → %s", len(rows), artifact_path)
    except Exception as exc:
        log.warning("Calibration curve computation failed: %s", exc)
