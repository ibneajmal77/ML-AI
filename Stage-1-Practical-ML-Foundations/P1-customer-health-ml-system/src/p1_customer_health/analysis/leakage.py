from __future__ import annotations

from pathlib import Path

import pandas as pd

from p1_customer_health.domain.dataset import CLASSIFICATION_TARGET, REGRESSION_TARGET, TIMESTAMP_COLUMN
from p1_customer_health.utils import ensure_dir, write_json


FORBIDDEN_COLUMNS = [
    "renewal_outcome",
    "actual_retention_call_result",
    "future_invoice_status",
]


def write_leakage_report(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    payload = {
        "chronology_ok": bool(train_df[TIMESTAMP_COLUMN].max() <= validation_df[TIMESTAMP_COLUMN].min() <= test_df[TIMESTAMP_COLUMN].min()),
        "forbidden_columns_present": [column for column in FORBIDDEN_COLUMNS if column in train_df.columns],
        "target_columns_in_feature_space": [column for column in [CLASSIFICATION_TARGET, REGRESSION_TARGET] if column in FORBIDDEN_COLUMNS],
        "split_summary": {
            "train_max_date": str(train_df[TIMESTAMP_COLUMN].max().date()),
            "validation_min_date": str(validation_df[TIMESTAMP_COLUMN].min().date()),
            "validation_max_date": str(validation_df[TIMESTAMP_COLUMN].max().date()),
            "test_min_date": str(test_df[TIMESTAMP_COLUMN].min().date()),
        },
    }
    write_json(output_dir / "leakage_report.json", payload)
