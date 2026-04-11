from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from world_cricket_ml.training.metrics import classification_metrics, regression_metrics
from world_cricket_ml.training.preprocessing import FEATURE_COLUMNS, build_preprocessor
from world_cricket_ml.training.splits import time_split
from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def run_boosting_benchmark(df: pd.DataFrame, artifact_root: Path) -> None:
    """Benchmark XGBoost and LightGBM against the baseline sklearn models.

    Both classification and regression tasks are benchmarked so the comparison
    is symmetric with training/classification.py and training/regression.py.

    Install with: pip install -e ".[boosting]"
    """
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        log.info("Boosting benchmark skipped — xgboost/lightgbm not installed.")
        write_json(
            artifact_root / "boosting" / "status.json",
            {
                "status": "skipped",
                "reason": "pip install world-cricket-ml-foundations[boosting]",
            },
        )
        return

    frame = df[df["future_matches_available"] >= 3].copy()
    train_df, test_df = time_split(frame)
    log.info("Boosting benchmark: train=%d  test=%d", len(train_df), len(test_df))

    classifier_candidates = {
        "xgboost_classifier": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
        "lightgbm_classifier": LGBMClassifier(
            n_estimators=250, learning_rate=0.05, random_state=42, verbose=-1
        ),
    }
    classification_results: dict[str, dict] = {}
    for name, estimator in classifier_candidates.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["dominant_next_cycle"])
        probabilities = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        classification_results[name] = classification_metrics(
            test_df["dominant_next_cycle"], probabilities, 0.5
        )
        log.info(
            "  %-30s  roc_auc=%.3f  f1=%.3f",
            name,
            classification_results[name]["roc_auc"],
            classification_results[name]["f1"],
        )

    regressor_candidates = {
        "xgboost_regressor": XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        ),
        "lightgbm_regressor": LGBMRegressor(
            n_estimators=250, learning_rate=0.05, random_state=42, verbose=-1
        ),
    }
    regression_results: dict[str, dict] = {}
    for name, estimator in regressor_candidates.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["future_win_rate_next_5"])
        predictions = pipeline.predict(test_df[FEATURE_COLUMNS])
        regression_results[name] = regression_metrics(test_df["future_win_rate_next_5"], predictions)
        log.info(
            "  %-30s  r2=%.3f  mae=%.4f  mape=%s",
            name,
            regression_results[name]["r2"],
            regression_results[name]["mae"],
            f"{regression_results[name]['mape']:.4f}"
            if regression_results[name]["mape"] is not None
            else "n/a",
        )

    write_json(
        artifact_root / "boosting" / "status.json",
        {
            "status": "completed",
            "classification_metrics": classification_results,
            "regression_metrics": regression_results,
        },
    )
    log.info("Boosting benchmark complete.")
