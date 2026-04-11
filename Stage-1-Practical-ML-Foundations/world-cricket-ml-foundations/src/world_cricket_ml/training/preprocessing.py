from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "won_toss",
    "batted_first",
    "team_runs",
    "team_wickets_lost",
    "opponent_runs",
    "opponent_wickets_lost",
    "team_run_rate",
    "opponent_run_rate",
    "match_nrr",
    "rolling_3_win_rate",
    "rolling_5_win_rate",
    "rolling_10_win_rate",
    "rolling_3_avg_nrr",
    "rolling_5_avg_nrr",
    "rolling_10_avg_nrr",
    "rolling_3_avg_runs",
    "rolling_5_avg_runs",
    "rolling_10_avg_runs",
    "matches_played_so_far",
    "rest_days",
    "opponent_rolling_5_win_rate",
    "year",
    "month",
    "quarter",
]

CATEGORICAL_FEATURES = ["team", "opponent", "venue", "city", "tournament", "toss_decision", "season"]

# Single source of truth for the full feature contract used by training AND serving.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
