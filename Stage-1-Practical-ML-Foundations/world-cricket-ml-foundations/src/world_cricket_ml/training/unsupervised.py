from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from world_cricket_ml.training.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, build_preprocessor
from world_cricket_ml.utils import ensure_dir

log = logging.getLogger(__name__)


def run_unsupervised(latest: pd.DataFrame, artifact_root: Path) -> pd.DataFrame:
    ensure_dir(artifact_root / "unsupervised")
    log.info("Running unsupervised: clustering + anomaly detection on %d teams", len(latest))

    cluster_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", KMeans(n_clusters=4, random_state=42, n_init=10)),
    ])
    # Pass only the feature columns — never the full frame which includes targets and signals.
    cluster_pipeline.fit(latest[FEATURE_COLUMNS])
    latest = latest.copy()
    latest["cluster_id"] = cluster_pipeline.predict(latest[FEATURE_COLUMNS])
    log.debug("Cluster distribution: %s", latest["cluster_id"].value_counts().to_dict())

    anomaly_model = IsolationForest(contamination=0.15, random_state=42)
    anomaly_model.fit(latest[NUMERIC_FEATURES])
    latest["anomaly_score"] = anomaly_model.decision_function(latest[NUMERIC_FEATURES])
    latest["is_anomaly"] = (anomaly_model.predict(latest[NUMERIC_FEATURES]) == -1).astype(int)
    n_anomalies = int(latest["is_anomaly"].sum())
    log.info("Anomaly detection: %d/%d teams flagged", n_anomalies, len(latest))

    latest.to_csv(artifact_root / "unsupervised" / "team_segmentation.csv", index=False)
    latest[latest["is_anomaly"] == 1].to_csv(artifact_root / "unsupervised" / "anomaly_detection.csv", index=False)
    joblib.dump(cluster_pipeline, artifact_root / "unsupervised" / "segmenter.joblib")
    log.info("Unsupervised artifacts saved to %s", artifact_root / "unsupervised")

    # Suppress unused-import warning; CATEGORICAL_FEATURES kept for callers that import from here.
    _ = CATEGORICAL_FEATURES
    return latest
