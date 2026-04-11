from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from world_cricket_ml.training.metrics import classification_metrics
from world_cricket_ml.training.splits import time_split
from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def run_self_supervised_proxy(df: pd.DataFrame, artifact_root: Path) -> None:
    """Text-feature proxy experiment.

    Uses TF-IDF + SVD to build dense embeddings from raw match narratives, then
    trains a supervised classifier on top.  The SVD step acts as an unsupervised
    representation layer (analogous to a self-supervised pre-training stage), but
    the downstream LogisticRegression is fully supervised — labels ARE used.
    This is therefore a *proxy* for self-supervised learning, not true SSL.
    """
    train = df[df["future_matches_available"] >= 3].copy()
    train_df, test_df = time_split(train)
    log.info("Self-supervised proxy: train=%d  test=%d", len(train_df), len(test_df))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("svd", TruncatedSVD(n_components=12, random_state=42)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(train_df["match_text"], train_df["dominant_next_cycle"])
    probabilities = pipeline.predict_proba(test_df["match_text"])[:, 1]
    metrics = classification_metrics(test_df["dominant_next_cycle"], probabilities, 0.5)
    log.info("  roc_auc=%.3f  f1=%.3f", metrics["roc_auc"], metrics["f1"])

    payload = {
        "status": "completed",
        "method": (
            "Text-feature proxy: TF-IDF + SVD embeddings from match narratives, "
            "then supervised LogisticRegression.  The SVD stage is unsupervised "
            "(representation learning without labels); the classifier is supervised. "
            "This is NOT true self-supervised learning — it is included as a "
            "paradigm-contrast experiment."
        ),
        "metrics": metrics,
    }
    write_json(artifact_root / "self_supervised" / "status.json", payload)
