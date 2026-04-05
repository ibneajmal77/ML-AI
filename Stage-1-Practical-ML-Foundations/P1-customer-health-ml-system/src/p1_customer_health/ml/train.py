from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from p1_customer_health.ml.features import (
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    NUMERIC_FEATURES,
    REGRESSION_TARGET,
    TEXT_FEATURE,
    load_dataset,
    time_split,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mixed_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("txt", TfidfVectorizer(max_features=200), TEXT_FEATURE),
        ]
    )


def _dense_tabular_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def _classification_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "brier": round(float(brier_score_loss(y_true, scores)), 4),
    }


def _best_threshold(y_true: pd.Series, scores: np.ndarray, false_positive_cost: float = 1.0, false_negative_cost: float = 8.0) -> float:
    best_threshold = 0.5
    best_cost = float("inf")
    for threshold in np.linspace(0.1, 0.9, 33):
        preds = (scores >= threshold).astype(int)
        fp = float(((preds == 1) & (y_true == 0)).sum())
        fn = float(((preds == 0) & (y_true == 1)).sum())
        cost = fp * false_positive_cost + fn * false_negative_cost
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return round(best_threshold, 3)


def _classification_slice_report(df: pd.DataFrame, y_true: pd.Series, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    preds = (scores >= threshold).astype(int)
    report_rows: list[dict[str, Any]] = []
    for column in ["plan_type", "region", "industry"]:
        for group_value, group_df in df.groupby(column):
            idx = group_df.index
            group_true = y_true.loc[idx]
            group_preds = preds[idx]
            report_rows.append(
                {
                    "slice_column": column,
                    "slice_value": group_value,
                    "count": int(len(idx)),
                    "positive_rate": round(float(group_true.mean()), 4),
                    "predicted_positive_rate": round(float(group_preds.mean()), 4),
                    "recall": round(float(recall_score(group_true, group_preds, zero_division=0)), 4),
                    "precision": round(float(precision_score(group_true, group_preds, zero_division=0)), 4),
                }
            )
    return pd.DataFrame(report_rows)


def _calibration_report(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=8, strategy="quantile")
    return pd.DataFrame(
        {
            "mean_predicted_probability": np.round(mean_pred, 4),
            "observed_positive_rate": np.round(frac_pos, 4),
        }
    )


def train_classifier(df: pd.DataFrame, artifact_root: Path) -> None:
    split = time_split(df)
    output_dir = artifact_root / "classification"
    _ensure_dir(output_dir)

    train_df = split.train
    val_df = split.validation
    test_df = split.test
    y_train = train_df[CLASSIFICATION_TARGET]
    y_val = val_df[CLASSIFICATION_TARGET]
    y_test = test_df[CLASSIFICATION_TARGET]

    models: dict[str, Pipeline] = {
        "logistic_mixed": Pipeline(
            [
                ("preprocess", _mixed_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest_tabular": Pipeline(
            [
                ("preprocess", _dense_tabular_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ]
        ),
        "gradient_boosting_tabular": Pipeline(
            [
                ("preprocess", _dense_tabular_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    leaderboard: list[dict[str, Any]] = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_threshold = 0.5
    best_score = -1.0

    for name, pipeline in models.items():
        pipeline.fit(train_df, y_train)
        train_scores = pipeline.predict_proba(train_df)[:, 1]
        val_scores = pipeline.predict_proba(val_df)[:, 1]
        threshold = _best_threshold(y_val, val_scores)
        train_metrics = _classification_metrics(y_train, train_scores, threshold)
        val_metrics = _classification_metrics(y_val, val_scores, threshold)
        leaderboard.append(
            {
                "model": name,
                "threshold": threshold,
                "train_pr_auc": train_metrics["pr_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "val_precision": val_metrics["precision"],
                "val_brier": val_metrics["brier"],
            }
        )
        if val_metrics["pr_auc"] > best_score:
            best_score = val_metrics["pr_auc"]
            best_name = name
            best_pipeline = pipeline
            best_threshold = threshold

    assert best_pipeline is not None

    test_scores = best_pipeline.predict_proba(test_df)[:, 1]
    test_metrics = _classification_metrics(y_test, test_scores, best_threshold)

    calibration_df = _calibration_report(y_test, test_scores)
    calibration_df.to_csv(output_dir / "calibration_bins.csv", index=False)

    slice_df = _classification_slice_report(test_df, y_test, test_scores, best_threshold)
    slice_df.to_csv(output_dir / "slice_analysis.csv", index=False)

    metrics = {
        "selected_model": best_name,
        "selected_threshold": best_threshold,
        "leaderboard": leaderboard,
        "test_metrics": test_metrics,
    }
    _write_json(output_dir / "metrics.json", metrics)
    joblib.dump({"model": best_pipeline, "threshold": best_threshold, "task": "classification"}, output_dir / "model.joblib")


def train_regressor(df: pd.DataFrame, artifact_root: Path) -> None:
    split = time_split(df)
    output_dir = artifact_root / "regression"
    _ensure_dir(output_dir)

    train_df = split.train
    val_df = split.validation
    test_df = split.test
    y_train = train_df[REGRESSION_TARGET]
    y_val = val_df[REGRESSION_TARGET]
    y_test = test_df[REGRESSION_TARGET]

    models: dict[str, Pipeline] = {
        "ridge_mixed": Pipeline(
            [
                ("preprocess", _mixed_preprocessor()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_regressor": Pipeline(
            [
                ("preprocess", _dense_tabular_preprocessor()),
                ("model", RandomForestRegressor(n_estimators=250, random_state=42)),
            ]
        ),
    }

    leaderboard: list[dict[str, Any]] = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_mae = float("inf")

    for name, pipeline in models.items():
        pipeline.fit(train_df, y_train)
        train_preds = pipeline.predict(train_df)
        val_preds = pipeline.predict(val_df)
        row = {
            "model": name,
            "train_mae": round(float(mean_absolute_error(y_train, train_preds)), 4),
            "val_mae": round(float(mean_absolute_error(y_val, val_preds)), 4),
            "val_rmse": round(float(np.sqrt(mean_squared_error(y_val, val_preds))), 4),
            "val_r2": round(float(r2_score(y_val, val_preds)), 4),
        }
        leaderboard.append(row)
        if row["val_mae"] < best_mae:
            best_mae = row["val_mae"]
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None
    test_preds = best_pipeline.predict(test_df)
    metrics = {
        "selected_model": best_name,
        "leaderboard": leaderboard,
        "test_metrics": {
            "mae": round(float(mean_absolute_error(y_test, test_preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
            "r2": round(float(r2_score(y_test, test_preds)), 4),
        },
    }
    _write_json(output_dir / "metrics.json", metrics)
    joblib.dump({"model": best_pipeline, "task": "regression"}, output_dir / "model.joblib")


def train_unsupervised(df: pd.DataFrame, artifact_root: Path) -> None:
    output_dir = artifact_root / "unsupervised"
    _ensure_dir(output_dir)

    preprocessor = _dense_tabular_preprocessor()
    feature_matrix = preprocessor.fit_transform(df)

    segmenter = KMeans(n_clusters=4, random_state=42, n_init="auto")
    clusters = segmenter.fit_predict(feature_matrix)
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    anomaly_flags = anomaly_detector.fit_predict(feature_matrix)
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(feature_matrix)

    summary = pd.DataFrame(
        {
            "account_id": df["account_id"],
            "segment_id": clusters,
            "anomaly_flag": (anomaly_flags == -1).astype(int),
            "pca_1": np.round(reduced[:, 0], 4),
            "pca_2": np.round(reduced[:, 1], 4),
        }
    )
    summary.to_csv(output_dir / "customer_segments.csv", index=False)
    joblib.dump({"preprocessor": preprocessor, "model": segmenter, "task": "clustering"}, output_dir / "segmenter.joblib")
    joblib.dump({"preprocessor": preprocessor, "model": anomaly_detector, "task": "anomaly"}, output_dir / "anomaly_detector.joblib")


def run_optional_self_supervised(df: pd.DataFrame, artifact_root: Path) -> None:
    output_dir = artifact_root / "self_supervised"
    _ensure_dir(output_dir)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _write_json(
            output_dir / "status.json",
            {
                "status": "skipped",
                "reason": "sentence-transformers not installed",
                "what_this_would_do": "encode support_note text with a pretrained self-supervised embedding model and train a churn baseline on embeddings",
            },
        )
        return

    split = time_split(df)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    train_embeddings = encoder.encode(split.train[TEXT_FEATURE].tolist())
    test_embeddings = encoder.encode(split.test[TEXT_FEATURE].tolist())
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train_embeddings, split.train[CLASSIFICATION_TARGET])
    scores = model.predict_proba(test_embeddings)[:, 1]
    metrics = _classification_metrics(split.test[CLASSIFICATION_TARGET], scores, threshold=0.5)
    _write_json(output_dir / "status.json", {"status": "completed", "test_metrics": metrics})


def run_optional_boosting(df: pd.DataFrame, artifact_root: Path) -> None:
    output_dir = artifact_root / "boosting"
    _ensure_dir(output_dir)

    split = time_split(df)
    train_df = split.train
    test_df = split.test
    y_train = train_df[CLASSIFICATION_TARGET]
    y_test = split.test[CLASSIFICATION_TARGET]
    preprocessor = _dense_tabular_preprocessor()
    x_train = preprocessor.fit_transform(train_df)
    x_test = preprocessor.transform(test_df)

    candidates: list[tuple[str, Any]] = []
    try:
        from xgboost import XGBClassifier

        candidates.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                ),
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        candidates.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                ),
            )
        )
    except ImportError:
        pass

    if not candidates:
        _write_json(
            output_dir / "status.json",
            {
                "status": "skipped",
                "reason": "xgboost and lightgbm not installed",
            },
        )
        return

    leaderboard: list[dict[str, Any]] = []
    for name, model in candidates:
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        threshold = _best_threshold(y_test, scores)
        metrics = _classification_metrics(y_test, scores, threshold)
        leaderboard.append({"model": name, "threshold": threshold, "test_metrics": metrics})

    _write_json(output_dir / "status.json", {"status": "completed", "leaderboard": leaderboard})


def train_all(dataset_path: Path, artifact_root: Path) -> None:
    df = load_dataset(dataset_path)
    _ensure_dir(artifact_root)
    train_classifier(df, artifact_root)
    train_regressor(df, artifact_root)
    train_unsupervised(df, artifact_root)
    run_optional_self_supervised(df, artifact_root)
    run_optional_boosting(df, artifact_root)
