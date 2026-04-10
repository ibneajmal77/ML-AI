from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from p1_customer_health.analysis.failure_taxonomy import write_failure_taxonomy
from p1_customer_health.domain.dataset import CLASSIFICATION_TARGET, time_split
from p1_customer_health.training.metrics import (
    best_threshold,
    calibration_report,
    classification_metrics,
    classification_slice_report,
    ensure_dir,
    fit_diagnosis_report,
    random_resample,
    write_json,
)
from p1_customer_health.training.preprocessing import dense_tabular_preprocessor, mixed_preprocessor


def train_classifier(df: pd.DataFrame, artifact_root: Path) -> None:
    split = time_split(df)
    output_dir = artifact_root / "classification"
    ensure_dir(output_dir)
    train_df = split.train
    val_df = split.validation
    test_df = split.test
    y_train = train_df[CLASSIFICATION_TARGET]
    y_val = val_df[CLASSIFICATION_TARGET]
    y_test = test_df[CLASSIFICATION_TARGET]

    resampled_train = random_resample(train_df, CLASSIFICATION_TARGET, "oversample")
    undersampled_train = random_resample(train_df, CLASSIFICATION_TARGET, "undersample")
    models: dict[str, Pipeline] = {
        "logistic_mixed": Pipeline([("preprocess", mixed_preprocessor()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "random_forest_tabular": Pipeline([("preprocess", dense_tabular_preprocessor()), ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))]),
        "gradient_boosting_tabular": Pipeline([("preprocess", dense_tabular_preprocessor()), ("model", GradientBoostingClassifier(random_state=42))]),
        "logistic_oversampled": Pipeline([("preprocess", mixed_preprocessor()), ("model", LogisticRegression(max_iter=1000))]),
        "logistic_undersampled": Pipeline([("preprocess", mixed_preprocessor()), ("model", LogisticRegression(max_iter=1000))]),
    }
    training_frames = {
        "logistic_mixed": train_df,
        "random_forest_tabular": train_df,
        "gradient_boosting_tabular": train_df,
        "logistic_oversampled": resampled_train,
        "logistic_undersampled": undersampled_train,
    }

    leaderboard: list[dict[str, Any]] = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_threshold_value = 0.5
    best_score = -1.0
    for name, pipeline in models.items():
        model_train_df = training_frames[name]
        model_y_train = model_train_df[CLASSIFICATION_TARGET]
        pipeline.fit(model_train_df, model_y_train)
        train_scores = pipeline.predict_proba(train_df)[:, 1]
        val_scores = pipeline.predict_proba(val_df)[:, 1]
        threshold = best_threshold(y_val, val_scores)
        train_metrics = classification_metrics(y_train, train_scores, threshold)
        val_metrics = classification_metrics(y_val, val_scores, threshold)
        leaderboard.append({"model": name, "threshold": threshold, "train_pr_auc": train_metrics["pr_auc"], "val_pr_auc": val_metrics["pr_auc"], "train_recall": train_metrics["recall"], "val_recall": val_metrics["recall"], "val_precision": val_metrics["precision"], "val_brier": val_metrics["brier"]})
        if val_metrics["pr_auc"] > best_score:
            best_score = val_metrics["pr_auc"]
            best_name = name
            best_pipeline = pipeline
            best_threshold_value = threshold

    assert best_pipeline is not None

    # Calibrate on training data, tune threshold on validation (never on test)
    calibrated_pipeline = CalibratedClassifierCV(clone(best_pipeline), method="sigmoid", cv=3)
    calibrated_pipeline.fit(train_df, y_train)
    calibrated_val_scores = calibrated_pipeline.predict_proba(val_df)[:, 1]
    calibrated_threshold = best_threshold(y_val, calibrated_val_scores)

    # Final evaluation on held-out test set — no decisions made from this
    test_scores = best_pipeline.predict_proba(test_df)[:, 1]
    calibrated_test_scores = calibrated_pipeline.predict_proba(test_df)[:, 1]
    test_metrics = classification_metrics(y_test, test_scores, best_threshold_value)
    calibrated_metrics = classification_metrics(y_test, calibrated_test_scores, calibrated_threshold)

    calibration_report(y_test, test_scores).to_csv(output_dir / "calibration_bins.csv", index=False)
    calibration_report(y_test, calibrated_test_scores).to_csv(output_dir / "calibrated_calibration_bins.csv", index=False)
    classification_slice_report(test_df, y_test, test_scores, best_threshold_value).to_csv(output_dir / "slice_analysis.csv", index=False)
    write_failure_taxonomy(test_df, test_scores, best_threshold_value, output_dir)
    fit_diagnosis_report(leaderboard, output_dir)
    write_json(output_dir / "metrics.json", {"selected_model": best_name, "selected_threshold": best_threshold_value, "selected_calibrated_threshold": calibrated_threshold, "leaderboard": leaderboard, "test_metrics": test_metrics, "calibrated_test_metrics": calibrated_metrics})

    trained_at = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    joblib.dump({"model": best_pipeline, "threshold": best_threshold_value, "task": "classification", "trained_at": trained_at}, output_dir / "model.joblib")
    joblib.dump({"model": calibrated_pipeline, "threshold": calibrated_threshold, "task": "classification_calibrated", "trained_at": trained_at}, output_dir / "calibrated_model.joblib")
