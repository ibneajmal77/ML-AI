from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from world_cricket_ml.analysis.failure_taxonomy import build_failure_taxonomy
from world_cricket_ml.training.metrics import classification_metrics, save_calibration_curve
from world_cricket_ml.training.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
)
from world_cricket_ml.training.splits import time_split
from world_cricket_ml.utils import ensure_dir, write_json

log = logging.getLogger(__name__)

__all__ = ["FEATURE_COLUMNS", "train_classifier"]

_ = NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Optional: imbalanced-learn for SMOTE (pip install -e ".[imbalanced]")
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _IMBLEARN_AVAILABLE = True
except ImportError:
    _IMBLEARN_AVAILABLE = False


def _oversample_positive_class(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    positives = df[df[target_column] == 1]
    negatives = df[df[target_column] == 0]
    if positives.empty or len(positives) >= len(negatives):
        return df
    sampled = positives.sample(len(negatives) - len(positives), replace=True, random_state=42)
    return pd.concat([df, sampled], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def _save_feature_importance(pipeline, artifact_dir: Path) -> None:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_type = "gini_impurity_reduction"
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
            importance_type = "abs_coefficient"
        else:
            return

        importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = importance_df.index + 1
        importance_df["importance_type"] = importance_type
        importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
        log.info(
            "Feature importance saved (type=%s) — top: %s (%.4f)",
            importance_type,
            importance_df["feature"].iloc[0],
            importance_df["importance"].iloc[0],
        )
    except Exception as exc:
        log.warning("Feature importance extraction failed: %s", exc)


def _build_pipeline(name: str, estimator) -> tuple[Pipeline, bool]:
    if name.startswith("smote_") and _IMBLEARN_AVAILABLE:
        return ImbPipeline([
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42)),
            ("model", estimator),
        ]), True
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ]), False


def train_classifier(df: pd.DataFrame, artifact_root: Path) -> tuple[Pipeline, pd.DataFrame]:
    ensure_dir(artifact_root / "classification")
    df = df[df["future_matches_available"] >= 3].copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    train_df, test_df = time_split(df)
    sampled_train = _oversample_positive_class(train_df, "dominant_next_cycle")
    log.info(
        "Classification split: train=%d (sampled=%d) test=%d  positive_rate=%.1f%%",
        len(train_df),
        len(sampled_train),
        len(test_df),
        100 * train_df["dominant_next_cycle"].mean(),
    )

    models: dict[str, object] = {
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "logistic_regression_weighted": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42),
        "random_forest_weighted": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    if _IMBLEARN_AVAILABLE:
        models["smote_random_forest"] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=42
        )

    metrics_payload: dict[str, dict] = {}
    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None

    for name, estimator in models.items():
        pipeline, uses_smote = _build_pipeline(name, estimator)

        # SMOTE and class_weight variants operate on the raw imbalanced frame;
        # they handle minority overrepresentation internally.
        fit_frame = train_df if (uses_smote or "weighted" in name) else sampled_train
        pipeline.fit(fit_frame[FEATURE_COLUMNS], fit_frame["dominant_next_cycle"])

        probabilities = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

        # Sweep thresholds across the full range to expose precision/recall tradeoffs
        # at the extremes, not just around 0.5.
        thresholds = np.linspace(0.1, 0.9, 17)
        evaluated = []
        for threshold in thresholds:
            current_metrics = classification_metrics(test_df["dominant_next_cycle"], probabilities, threshold)
            cost = (
                ((test_df["dominant_next_cycle"] == 1) & (probabilities < threshold)).sum() * 5
                + ((test_df["dominant_next_cycle"] == 0) & (probabilities >= threshold)).sum() * 2
            )
            current_metrics["business_cost"] = int(cost)
            evaluated.append(current_metrics)
        selected = sorted(evaluated, key=lambda item: (item["business_cost"], -item["f1"]))[0]

        train_probs = pipeline.predict_proba(fit_frame[FEATURE_COLUMNS])[:, 1]
        train_roc = classification_metrics(
            fit_frame["dominant_next_cycle"], train_probs, selected["threshold"]
        )["roc_auc"]
        selected["train_roc_auc"] = train_roc

        metrics_payload[name] = selected
        log.info(
            "  %-30s  test_roc=%.3f  train_roc=%.3f  f1=%.3f  threshold=%.2f  cost=%d",
            name,
            selected["roc_auc"],
            train_roc,
            selected["f1"],
            selected["threshold"],
            selected["business_cost"],
        )
        if selected["f1"] > best_score:
            best_score = selected["f1"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise ValueError("No pipeline was trained — models dict is empty.")

    log.info("Best classifier: %s (test F1=%.3f)", best_name, best_score)

    # Time-aware cross-validation on the best model.
    # TimeSeriesSplit prevents future information from leaking into validation folds.
    try:
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(
            best_pipeline,
            train_df[FEATURE_COLUMNS],
            train_df["dominant_next_cycle"],
            cv=tscv,
            scoring="roc_auc",
        )
        cv_summary = {
            "method": "TimeSeriesSplit(n_splits=3)",
            "cv_roc_auc_scores": [round(float(s), 4) for s in cv_scores],
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        }
        log.info(
            "Cross-validation (TimeSeriesSplit n=3): roc_auc=%.3f ± %.3f",
            cv_scores.mean(),
            cv_scores.std(),
        )
    except Exception as exc:
        log.warning("Cross-validation failed (dataset may be too small): %s", exc)
        cv_summary = {
            "method": "TimeSeriesSplit(n_splits=3)",
            "cv_roc_auc_scores": [],
            "cv_roc_auc_mean": None,
            "cv_roc_auc_std": None,
            "error": str(exc),
        }

    raw_probs = best_pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    calibrated = CalibratedClassifierCV(best_pipeline, method="sigmoid", cv=3)
    calibrated.fit(train_df[FEATURE_COLUMNS], train_df["dominant_next_cycle"])
    calibrated_probs = calibrated.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    selected_threshold = metrics_payload[best_name]["threshold"]
    calibration_test_metrics = classification_metrics(
        test_df["dominant_next_cycle"], calibrated_probs, selected_threshold
    )
    selected_artifact_name = (
        "calibrated_model"
        if calibration_test_metrics["roc_auc"] >= metrics_payload[best_name]["roc_auc"]
        else "raw_model"
    )
    log.info("Serving artifact: %s", selected_artifact_name)

    serving_probs = calibrated_probs if selected_artifact_name == "calibrated_model" else raw_probs
    scored = test_df.copy()
    scored["dominance_probability"] = serving_probs
    scored["predicted_label"] = (scored["dominance_probability"] >= selected_threshold).astype(int)
    scored["prediction_error"] = scored["dominant_next_cycle"] - scored["predicted_label"]
    scored.to_csv(artifact_root / "classification" / "predictions.csv", index=False)

    slice_analysis = (
        scored.assign(
            form_bucket=pd.cut(
                scored["rolling_5_win_rate"],
                bins=[-0.01, 0.35, 0.55, 1.0],
                labels=["low", "mid", "high"],
            )
        )
        .groupby(["team", "form_bucket"], observed=False)
        .agg(
            actual_rate=("dominant_next_cycle", "mean"),
            predicted_rate=("predicted_label", "mean"),
            rows=("match_id", "count"),
        )
        .reset_index()
    )
    slice_analysis.to_csv(artifact_root / "classification" / "slice_analysis.csv", index=False)

    fit_diagnosis = pd.DataFrame([
        {
            "model_name": name,
            "f1": payload["f1"],
            "roc_auc": payload["roc_auc"],
            "train_roc_auc": payload["train_roc_auc"],
            "train_test_gap": round(payload["train_roc_auc"] - payload["roc_auc"], 4),
            "business_cost": payload["business_cost"],
            "diagnosis": (
                "overfitting_risk"
                if payload["train_roc_auc"] - payload["roc_auc"] > 0.15
                else "underfitting_risk"
                if payload["roc_auc"] < 0.55 and payload["train_roc_auc"] < 0.60
                else "healthy"
            ),
        }
        for name, payload in metrics_payload.items()
    ])
    fit_diagnosis.to_csv(artifact_root / "classification" / "fit_diagnosis.csv", index=False)
    build_failure_taxonomy(scored, artifact_root)

    _save_feature_importance(best_pipeline, artifact_root / "classification")

    save_calibration_curve(
        test_df["dominant_next_cycle"],
        raw_probs,
        calibrated_probs,
        artifact_root / "classification" / "calibration_curve.csv",
    )

    write_json(artifact_root / "classification" / "metrics.json", {
        "best_model": best_name,
        "candidate_metrics": metrics_payload,
        "calibrated_model_metrics": calibration_test_metrics,
        "selected_serving_artifact": selected_artifact_name,
        "serving_threshold": selected_threshold,
        "cross_validation": cv_summary,
        "smote_available": _IMBLEARN_AVAILABLE,
    })
    joblib.dump(best_pipeline, artifact_root / "classification" / "model.joblib")
    joblib.dump(calibrated, artifact_root / "classification" / "calibrated_model.joblib")
    log.info("Classification artifacts saved to %s", artifact_root / "classification")
    return calibrated if selected_artifact_name == "calibrated_model" else best_pipeline, scored
