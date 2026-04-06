from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _bucket_failure(row: pd.Series) -> str:
    if row["error_type"] == "false_negative":
        if row["days_since_last_activity"] > 25:
            return "missed_inactive_customer"
        if row["tickets_30d"] >= 5:
            return "missed_support_distress"
        if row["plan_type"] == "enterprise":
            return "missed_high_value_enterprise"
        return "other_false_negative"
    if row["error_type"] == "false_positive":
        if row["tickets_30d"] >= 5 and row["churned_30d"] == 0:
            return "support_noise_false_alarm"
        if row["feature_adoption_ratio"] > 0.7:
            return "healthy_usage_false_alarm"
        return "other_false_positive"
    return "correct_prediction"


def write_failure_taxonomy(df: pd.DataFrame, scores: np.ndarray, threshold: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = (scores >= threshold).astype(int)
    result = df.copy()
    result["predicted_label"] = preds
    result["error_type"] = np.where(
        (result["churned_30d"] == 1) & (result["predicted_label"] == 0),
        "false_negative",
        np.where(
            (result["churned_30d"] == 0) & (result["predicted_label"] == 1),
            "false_positive",
            "correct",
        ),
    )
    result["failure_bucket"] = result.apply(_bucket_failure, axis=1)
    result.to_csv(output_dir / "failure_taxonomy_rows.csv", index=False)
    (
        result.groupby(["error_type", "failure_bucket"]).size().reset_index(name="count")
        .sort_values(["error_type", "count"], ascending=[True, False])
        .to_csv(output_dir / "failure_taxonomy_summary.csv", index=False)
    )
