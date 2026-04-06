from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from p1_customer_health.domain.dataset import CATEGORICAL_FEATURES, CLASSIFICATION_TARGET, NUMERIC_FEATURES, REGRESSION_TARGET, TIMESTAMP_COLUMN


def write_data_quality_report(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "row_count": int(len(df)),
        "duplicate_account_snapshot_rows": int(df.duplicated(subset=["account_id", TIMESTAMP_COLUMN]).sum()),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "classification_positive_rate": round(float(df[CLASSIFICATION_TARGET].mean()), 4),
        "regression_summary": {
            "mean": round(float(df[REGRESSION_TARGET].mean()), 4),
            "std": round(float(df[REGRESSION_TARGET].std()), 4),
        },
        "numeric_ranges": {
            column: {
                "min": round(float(df[column].min()), 4),
                "max": round(float(df[column].max()), 4),
                "mean": round(float(df[column].mean()), 4),
            }
            for column in NUMERIC_FEATURES
        },
        "categorical_cardinality": {column: int(df[column].nunique()) for column in CATEGORICAL_FEATURES},
    }
    (output_dir / "data_quality_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
