from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from p1_customer_health.domain.dataset import REGRESSION_TARGET, time_split
from p1_customer_health.training.metrics import ensure_dir, fit_diagnosis_report, write_json
from p1_customer_health.training.preprocessing import dense_tabular_preprocessor, mixed_preprocessor


def train_regressor(df: pd.DataFrame, artifact_root: Path) -> None:
    split = time_split(df)
    output_dir = artifact_root / "regression"
    ensure_dir(output_dir)
    train_df = split.train
    val_df = split.validation
    test_df = split.test
    y_train = train_df[REGRESSION_TARGET]
    y_val = val_df[REGRESSION_TARGET]
    y_test = test_df[REGRESSION_TARGET]

    models: dict[str, Pipeline] = {
        "ridge_mixed": Pipeline([("preprocess", mixed_preprocessor()), ("model", Ridge(alpha=1.0))]),
        "random_forest_regressor": Pipeline([("preprocess", dense_tabular_preprocessor()), ("model", RandomForestRegressor(n_estimators=250, random_state=42))]),
    }

    leaderboard: list[dict[str, Any]] = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_mae = float("inf")
    for name, pipeline in models.items():
        pipeline.fit(train_df, y_train)
        train_preds = pipeline.predict(train_df)
        val_preds = pipeline.predict(val_df)
        row = {"model": name, "train_mae": round(float(mean_absolute_error(y_train, train_preds)), 4), "val_mae": round(float(mean_absolute_error(y_val, val_preds)), 4), "val_rmse": round(float(np.sqrt(mean_squared_error(y_val, val_preds))), 4), "val_r2": round(float(r2_score(y_val, val_preds)), 4)}
        leaderboard.append(row)
        if row["val_mae"] < best_mae:
            best_mae = row["val_mae"]
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None
    test_preds = best_pipeline.predict(test_df)
    write_json(output_dir / "metrics.json", {"selected_model": best_name, "leaderboard": leaderboard, "test_metrics": {"mae": round(float(mean_absolute_error(y_test, test_preds)), 4), "rmse": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4), "r2": round(float(r2_score(y_test, test_preds)), 4)}})
    fit_diagnosis_report(leaderboard, output_dir)
    trained_at = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    joblib.dump({"model": best_pipeline, "task": "regression", "trained_at": trained_at}, output_dir / "model.joblib")
