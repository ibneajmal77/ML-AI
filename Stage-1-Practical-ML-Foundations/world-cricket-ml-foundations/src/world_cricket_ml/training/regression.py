from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from world_cricket_ml.training.metrics import regression_metrics
from world_cricket_ml.training.preprocessing import FEATURE_COLUMNS, build_preprocessor
from world_cricket_ml.training.splits import time_split
from world_cricket_ml.utils import ensure_dir, write_json

log = logging.getLogger(__name__)


def _save_feature_importance(pipeline, artifact_dir: Path) -> None:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_type = "variance_reduction"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
            importance_type = "abs_coefficient"
        else:
            return

        importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = importance_df.index + 1
        importance_df["importance_type"] = importance_type
        importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
        log.info(
            "Regression feature importance saved (type=%s) — top: %s (%.4f)",
            importance_type,
            importance_df["feature"].iloc[0],
            importance_df["importance"].iloc[0],
        )
    except Exception as exc:
        log.warning("Regression feature importance extraction failed: %s", exc)


def train_regressor(df: pd.DataFrame, artifact_root: Path) -> tuple[Pipeline, pd.DataFrame]:
    ensure_dir(artifact_root / "regression")
    df = df[df["future_matches_available"] >= 3].copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    train_df, test_df = time_split(df)
    log.info("Regression split: train=%d  test=%d", len(train_df), len(test_df))

    estimators = {
        "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=42),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
    }
    metrics_payload: dict[str, dict] = {}
    best_name = ""
    best_mae = float("inf")
    best_pipeline: Pipeline | None = None
    scored = test_df.copy()

    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["future_win_rate_next_5"])

        test_predictions = pipeline.predict(test_df[FEATURE_COLUMNS])
        test_m = regression_metrics(test_df["future_win_rate_next_5"], test_predictions)

        train_predictions = pipeline.predict(train_df[FEATURE_COLUMNS])
        train_m = regression_metrics(train_df["future_win_rate_next_5"], train_predictions)

        metrics_payload[name] = {
            **test_m,
            "train_r2": train_m["r2"],
            "train_mae": train_m["mae"],
            "train_mape": train_m["mape"],
        }
        log.info(
            "  %-30s  test_r2=%.3f  train_r2=%.3f  test_mae=%.4f  test_mape=%s",
            name,
            test_m["r2"],
            train_m["r2"],
            test_m["mae"],
            f"{test_m['mape']:.4f}" if test_m["mape"] is not None else "n/a",
        )
        if test_m["mae"] < best_mae:
            best_mae = test_m["mae"]
            best_name = name
            best_pipeline = pipeline
            scored["predicted_future_win_rate"] = test_predictions

    if best_pipeline is None:
        raise ValueError("No pipeline was trained — estimators dict is empty.")

    log.info("Best regressor: %s (test MAE=%.4f)", best_name, best_mae)
    scored.to_csv(artifact_root / "regression" / "predictions.csv", index=False)

    fit_diagnosis = pd.DataFrame([
        {
            "model_name": name,
            "test_mae": payload["mae"],
            "test_rmse": payload["rmse"],
            "test_r2": payload["r2"],
            "test_mape": payload["mape"],
            "train_r2": payload["train_r2"],
            "train_test_r2_gap": round(payload["train_r2"] - payload["r2"], 4),
            "diagnosis": (
                "underfitting"
                if payload["r2"] < 0.10 and payload["train_r2"] < 0.15
                else "overfitting_risk"
                if payload["train_r2"] - payload["r2"] > 0.20
                else "healthy"
            ),
        }
        for name, payload in metrics_payload.items()
    ])
    fit_diagnosis.to_csv(artifact_root / "regression" / "fit_diagnosis.csv", index=False)
    write_json(artifact_root / "regression" / "metrics.json", {
        "best_model": best_name,
        "candidate_metrics": metrics_payload,
    })
    joblib.dump(best_pipeline, artifact_root / "regression" / "model.joblib")

    _save_feature_importance(best_pipeline, artifact_root / "regression")

    log.info("Regression artifacts saved to %s", artifact_root / "regression")
    return best_pipeline, scored
