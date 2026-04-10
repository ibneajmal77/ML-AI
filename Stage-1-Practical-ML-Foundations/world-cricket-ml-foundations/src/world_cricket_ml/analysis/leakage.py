from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def build_leakage_report(df: pd.DataFrame, feature_columns: list[str], artifact_root: Path) -> dict:
    """Detect and report potential data leakage in three stages.

    1. Name-pattern check: block columns whose names encode future information
       ("future_" prefix, "_next_cycle" suffix).
    2. Feature contract check: flag any contracted feature whose name contains
       leakage tokens ("future", "label", "target").
    3. Mutual information scan: compute MI between each numeric feature and the
       label.  A score above 0.5 indicates suspiciously high correlation that
       may encode future information in a non-obvious way.
    """
    blocked_columns = [
        column for column in df.columns
        if column.startswith("future_") or column.endswith("_next_cycle")
    ]
    suspicious = [
        column for column in feature_columns
        if any(token in column for token in ("future", "label", "target"))
    ]
    if suspicious:
        log.warning("Suspicious feature names detected: %s", suspicious)
    else:
        log.info("Leakage name-check passed — no suspicious feature names found.")
    log.info("Blocked future columns: %s", blocked_columns)

    mi_scan: dict[str, float] = {}
    high_mi_features: list[str] = []
    try:
        numeric_feature_cols = [
            c for c in feature_columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        label_col = "dominant_next_cycle"
        if numeric_feature_cols and label_col in df.columns:
            mi_data = df[numeric_feature_cols + [label_col]].dropna()
            if len(mi_data) > 20:
                mi_scores = mutual_info_classif(
                    mi_data[numeric_feature_cols],
                    mi_data[label_col],
                    random_state=42,
                )
                mi_scan = {
                    col: round(float(score), 4)
                    for col, score in zip(numeric_feature_cols, mi_scores)
                }
                high_mi_features = [col for col, score in mi_scan.items() if score > 0.5]
                if high_mi_features:
                    log.warning("High MI features (potential leakage, MI > 0.5): %s", high_mi_features)
                else:
                    log.info("Mutual information scan passed — no high-MI features found.")
            else:
                log.info("Mutual information scan skipped — insufficient rows (%d).", len(mi_data))
    except Exception as exc:
        log.warning("Mutual information scan failed: %s", exc)

    status = "review_required" if (suspicious or high_mi_features) else "ok"
    payload = {
        "blocked_from_training": sorted(blocked_columns),
        "suspicious_feature_names": suspicious,
        "time_split_policy": (
            "train on older dates and evaluate on the newest 20 percent of rows by match date"
        ),
        "implemented_holdout": "single time-based holdout split",
        "status": status,
        "mutual_info_scan": mi_scan,
        "high_mi_features": high_mi_features,
    }
    write_json(artifact_root / "leakage" / "leakage_report.json", payload)
    return payload
