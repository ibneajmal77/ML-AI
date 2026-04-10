from __future__ import annotations

import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from p1_customer_health.training.metrics import ensure_dir
from p1_customer_health.training.preprocessing import dense_tabular_preprocessor


def train_unsupervised(df: pd.DataFrame, artifact_root: Path) -> None:
    output_dir = artifact_root / "unsupervised"
    ensure_dir(output_dir)
    preprocessor = dense_tabular_preprocessor()
    feature_matrix = preprocessor.fit_transform(df)
    segmenter = KMeans(n_clusters=4, random_state=42, n_init="auto")
    clusters = segmenter.fit_predict(feature_matrix)
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    anomaly_flags = anomaly_detector.fit_predict(feature_matrix)
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(feature_matrix)
    pd.DataFrame({"account_id": df["account_id"], "segment_id": clusters, "anomaly_flag": (anomaly_flags == -1).astype(int), "pca_1": np.round(reduced[:, 0], 4), "pca_2": np.round(reduced[:, 1], 4)}).to_csv(output_dir / "customer_segments.csv", index=False)
    trained_at = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    joblib.dump({"preprocessor": preprocessor, "model": segmenter, "task": "clustering", "trained_at": trained_at}, output_dir / "segmenter.joblib")
    joblib.dump({"preprocessor": preprocessor, "model": anomaly_detector, "task": "anomaly", "trained_at": trained_at}, output_dir / "anomaly_detector.joblib")
