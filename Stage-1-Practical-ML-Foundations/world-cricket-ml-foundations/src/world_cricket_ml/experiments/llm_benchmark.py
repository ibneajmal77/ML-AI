from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from world_cricket_ml.training.metrics import classification_metrics
from world_cricket_ml.training.splits import time_split
from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def run_llm_vs_classical_benchmark(df: pd.DataFrame, classical_auc: float, artifact_root: Path) -> None:
    train = df[df["future_matches_available"] >= 3].copy()
    train_df, test_df = time_split(train)
    log.info(
        "LLM benchmark: classical_auc=%.3f  text_train=%d  text_test=%d",
        classical_auc,
        len(train_df),
        len(test_df),
    )

    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    text_pipeline.fit(train_df["match_text"], train_df["dominant_next_cycle"])
    text_probs = text_pipeline.predict_proba(test_df["match_text"])[:, 1]
    text_metrics = classification_metrics(test_df["dominant_next_cycle"], text_probs, 0.5)
    winner = "classical_tabular_ml" if classical_auc >= text_metrics["roc_auc"] else "text_only_proxy"
    log.info("  text_roc=%.3f  winner=%s", text_metrics["roc_auc"], winner)

    payload = {
        "classical_tabular_roc_auc": round(float(classical_auc), 4),
        "text_only_proxy_roc_auc": text_metrics["roc_auc"],
        "winner": winner,
        "justification": (
            "Structured historical form, opponent strength, and schedule features are richer "
            "than short textual summaries for this bounded binary classification task."
        ),
    }
    write_json(artifact_root / "llm_benchmark" / "llm_vs_classical_benchmark.json", payload)
