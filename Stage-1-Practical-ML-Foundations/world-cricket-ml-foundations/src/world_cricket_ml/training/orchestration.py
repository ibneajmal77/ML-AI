from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from world_cricket_ml.analysis.audit import build_data_quality_report
from world_cricket_ml.analysis.business_framing import build_business_framing
from world_cricket_ml.analysis.leakage import build_leakage_report
from world_cricket_ml.domain.dataset import build_project_datasets
from world_cricket_ml.experiments.boosting import run_boosting_benchmark
from world_cricket_ml.experiments.llm_benchmark import run_llm_vs_classical_benchmark
from world_cricket_ml.experiments.onnx_export import export_onnx_stub
from world_cricket_ml.experiments.rl import run_contextual_bandit_proxy
from world_cricket_ml.experiments.self_supervised import run_self_supervised_proxy
from world_cricket_ml.training.classification import FEATURE_COLUMNS, train_classifier
from world_cricket_ml.training.regression import train_regressor
from world_cricket_ml.training.unsupervised import run_unsupervised
from world_cricket_ml.utils import compute_team_signals, read_json, write_json

log = logging.getLogger(__name__)


def train_all(project_root: Path) -> None:
    log.info("=== train_all start  root=%s ===", project_root)

    build_project_datasets(project_root)
    snapshots = pd.read_csv(
        project_root / "data" / "raw" / "world_cricket_snapshots.csv",
        parse_dates=["match_date"],
    )
    latest = pd.read_csv(
        project_root / "data" / "raw" / "world_cricket_latest_snapshots.csv",
        parse_dates=["match_date"],
    )
    log.info("Loaded snapshots=%d rows  latest=%d teams", len(snapshots), len(latest))

    build_data_quality_report(snapshots, project_root / "artifacts")
    build_leakage_report(snapshots, FEATURE_COLUMNS, project_root / "artifacts")

    classifier, scored_classification = train_classifier(snapshots, project_root / "artifacts")
    regressor, _regression_scores = train_regressor(snapshots, project_root / "artifacts")

    latest = latest.copy()
    latest["dominance_probability"] = classifier.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["predicted_future_win_rate"] = regressor.predict(latest[FEATURE_COLUMNS])
    latest = compute_team_signals(latest)

    run_unsupervised(latest, project_root / "artifacts")
    run_self_supervised_proxy(snapshots, project_root / "artifacts")
    run_contextual_bandit_proxy(snapshots, project_root / "artifacts")
    run_boosting_benchmark(snapshots, project_root / "artifacts")
    run_llm_vs_classical_benchmark(
        snapshots,
        classical_auc=float(
            roc_auc_score(
                scored_classification["dominant_next_cycle"],
                scored_classification["dominance_probability"],
            )
        ),
        artifact_root=project_root / "artifacts",
    )
    export_onnx_stub(project_root / "artifacts")
    build_business_framing(latest, project_root / "artifacts")

    _write_model_registry(project_root)

    log.info("=== train_all complete ===")


def _write_model_registry(project_root: Path) -> None:
    """Write a consolidated model registry artifact.

    Records the selected model variant, its test-set metrics, cross-validation
    stability, the data window, and a UTC training timestamp.  Overwrites on
    every retrain; git history provides the audit trail.
    """
    try:
        clf_json = read_json(project_root / "artifacts" / "classification" / "metrics.json")
        reg_json = read_json(project_root / "artifacts" / "regression" / "metrics.json")
        meta_json = read_json(project_root / "data" / "raw" / "dataset_metadata.json")

        best_clf = clf_json["best_model"]
        best_reg = reg_json["best_model"]
        clf_candidate = clf_json["candidate_metrics"].get(best_clf, {})
        reg_candidate = reg_json["candidate_metrics"].get(best_reg, {})

        registry = {
            "schema_version": "1.0",
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "data_window": meta_json.get("date_window", {}),
            "snapshot_rows": meta_json.get("snapshot_rows"),
            "teams_found": len(meta_json.get("teams_found", [])),
            "classification": {
                "best_model": best_clf,
                "serving_artifact": clf_json.get("selected_serving_artifact"),
                "serving_threshold": clf_json.get("serving_threshold"),
                "test_roc_auc": clf_candidate.get("roc_auc"),
                "test_f1": clf_candidate.get("f1"),
                "test_pr_auc": clf_candidate.get("pr_auc"),
                "test_business_cost": clf_candidate.get("business_cost"),
                "cross_validation": clf_json.get("cross_validation", {}),
            },
            "regression": {
                "best_model": best_reg,
                "test_mae": reg_candidate.get("mae"),
                "test_rmse": reg_candidate.get("rmse"),
                "test_r2": reg_candidate.get("r2"),
                "test_mape": reg_candidate.get("mape"),
            },
        }
        write_json(project_root / "artifacts" / "model_registry.json", registry)
        log.info(
            "Model registry written — clf=%s (roc_auc=%.3f)  reg=%s (mae=%.4f)",
            best_clf,
            clf_candidate.get("roc_auc", 0),
            best_reg,
            reg_candidate.get("mae", 0),
        )
    except Exception as exc:
        log.warning("Model registry write failed (non-fatal): %s", exc)
