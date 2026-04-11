from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from p1_customer_health.utils import ensure_dir, write_json


def classification_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "brier": round(float(brier_score_loss(y_true, scores)), 4),
    }


def best_threshold(y_true: pd.Series, scores: np.ndarray, false_positive_cost: float = 1.0, false_negative_cost: float = 8.0) -> float:
    winner = 0.5
    winner_cost = float("inf")
    for threshold in np.linspace(0.1, 0.9, 33):
        preds = (scores >= threshold).astype(int)
        fp = float(((preds == 1) & (y_true == 0)).sum())
        fn = float(((preds == 0) & (y_true == 1)).sum())
        cost = fp * false_positive_cost + fn * false_negative_cost
        if cost < winner_cost:
            winner = float(threshold)
            winner_cost = cost
    return round(winner, 3)


def classification_slice_report(df: pd.DataFrame, y_true: pd.Series, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    preds = (scores >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for column in ["plan_type", "region", "industry"]:
        for group_value, group_df in df.groupby(column):
            idx = group_df.index
            group_true = y_true.loc[idx]
            group_preds = preds[idx]
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": group_value,
                    "count": int(len(idx)),
                    "positive_rate": round(float(group_true.mean()), 4),
                    "predicted_positive_rate": round(float(group_preds.mean()), 4),
                    "recall": round(float(recall_score(group_true, group_preds, zero_division=0)), 4),
                    "precision": round(float(precision_score(group_true, group_preds, zero_division=0)), 4),
                }
            )
    return pd.DataFrame(rows)


def calibration_report(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=8, strategy="quantile")
    return pd.DataFrame(
        {
            "mean_predicted_probability": np.round(mean_pred, 4),
            "observed_positive_rate": np.round(frac_pos, 4),
        }
    )


def fit_diagnosis_report(leaderboard: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for row in leaderboard:
        gap = round(float(row.get("train_pr_auc", row.get("train_mae", 0)) - row.get("val_pr_auc", row.get("val_mae", 0))), 4)
        if "train_pr_auc" in row:
            status = "likely_overfitting" if gap > 0.08 else "stable_or_underfit"
        else:
            status = "likely_overfitting" if gap < -25 else "stable_or_underfit"
        rows.append({"model": row["model"], "gap": gap, "diagnosis": status})
    pd.DataFrame(rows).to_csv(output_dir / "fit_diagnosis.csv", index=False)


def random_resample(train_df: pd.DataFrame, target_column: str, strategy: str) -> pd.DataFrame:
    counts = train_df[target_column].value_counts()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    minority_df = train_df[train_df[target_column] == minority_class]
    majority_df = train_df[train_df[target_column] == majority_class]
    if strategy == "oversample":
        sampled_minority = minority_df.sample(len(majority_df), replace=True, random_state=42)
        return pd.concat([majority_df, sampled_minority], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    sampled_majority = majority_df.sample(len(minority_df), random_state=42)
    return pd.concat([sampled_majority, minority_df], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
