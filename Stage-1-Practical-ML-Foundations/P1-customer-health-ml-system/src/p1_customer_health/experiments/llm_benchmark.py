from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from p1_customer_health.domain.dataset import CLASSIFICATION_TARGET, TEXT_FEATURE, time_split


RISK_KEYWORDS = ["complaint", "frustration", "blocker", "low usage"]


def _heuristic_llm_style_label(text: str) -> int:
    lowered = text.lower()
    return int(any(keyword in lowered for keyword in RISK_KEYWORDS))


def run_llm_vs_classical_benchmark(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split = time_split(df)
    train_df = split.train
    test_df = split.test

    text_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=300)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    text_model.fit(train_df[TEXT_FEATURE], train_df[CLASSIFICATION_TARGET])
    classical_scores = text_model.predict_proba(test_df[TEXT_FEATURE])[:, 1]
    classical_preds = (classical_scores >= 0.5).astype(int)

    heuristic_preds = test_df[TEXT_FEATURE].map(_heuristic_llm_style_label).astype(int)
    payload = {
        "comparison_type": "classical_text_model_vs_llm_style_zero_shot_proxy",
        "classical_metrics": {
            "accuracy": round(float(accuracy_score(test_df[CLASSIFICATION_TARGET], classical_preds)), 4),
            "precision": round(float(precision_score(test_df[CLASSIFICATION_TARGET], classical_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(test_df[CLASSIFICATION_TARGET], classical_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(test_df[CLASSIFICATION_TARGET], classical_preds, zero_division=0)), 4),
            "pr_auc": round(float(average_precision_score(test_df[CLASSIFICATION_TARGET], classical_scores)), 4),
        },
        "llm_style_proxy_metrics": {
            "accuracy": round(float(accuracy_score(test_df[CLASSIFICATION_TARGET], heuristic_preds)), 4),
            "precision": round(float(precision_score(test_df[CLASSIFICATION_TARGET], heuristic_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(test_df[CLASSIFICATION_TARGET], heuristic_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(test_df[CLASSIFICATION_TARGET], heuristic_preds, zero_division=0)), 4),
        },
    }
    (output_dir / "llm_vs_classical_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
