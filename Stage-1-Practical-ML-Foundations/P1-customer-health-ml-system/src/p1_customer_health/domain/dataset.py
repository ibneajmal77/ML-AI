from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NUMERIC_FEATURES = [
    "monthly_spend",
    "logins_30d",
    "tickets_30d",
    "feature_adoption_ratio",
    "days_since_last_activity",
    "account_age_days",
    "nps_score",
    "payment_failures_90d",
    "seat_utilization",
]

CATEGORICAL_FEATURES = [
    "plan_type",
    "industry",
    "region",
    "contract_type",
]

TEXT_FEATURE = "support_note"
TIMESTAMP_COLUMN = "snapshot_date"
CLASSIFICATION_TARGET = "churned_30d"
REGRESSION_TARGET = "revenue_change_next_30d"


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_dataset(df: pd.DataFrame) -> None:
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE, TIMESTAMP_COLUMN, CLASSIFICATION_TARGET, REGRESSION_TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {sorted(missing)}")


def load_dataset(path: str | bytes | "os.PathLike[str]") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    validate_dataset(df)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.7, validation_frac: float = 0.15) -> DatasetSplit:
    ordered = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    train_end = int(len(ordered) * train_frac)
    validation_end = int(len(ordered) * (train_frac + validation_frac))
    return DatasetSplit(
        train=ordered.iloc[:train_end].copy().reset_index(drop=True),
        validation=ordered.iloc[train_end:validation_end].copy().reset_index(drop=True),
        test=ordered.iloc[validation_end:].copy().reset_index(drop=True),
    )
