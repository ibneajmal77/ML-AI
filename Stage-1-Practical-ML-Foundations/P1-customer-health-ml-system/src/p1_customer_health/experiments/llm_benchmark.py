from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from p1_customer_health.domain.dataset import CLASSIFICATION_TARGET, TEXT_FEATURE, time_split
from p1_customer_health.training.metrics import classification_metrics
from p1_customer_health.utils import ensure_dir, write_json


RISK_KEYWORDS = ["complaint", "frustration", "blocker", "low usage"]


def _heuristic_llm_style_label(text: str) -> int:
    lowered = text.lower()
    return int(any(keyword in lowered for keyword in RISK_KEYWORDS))


def run_llm_vs_classical_benchmark(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    split = time_split(df)
    train_df = split.train
    test_df = split.test
    y_test = test_df[CLASSIFICATION_TARGET]

    # Classical text model: TF-IDF + Logistic Regression trained on support notes
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=300)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    text_pipeline.fit(train_df[TEXT_FEATURE], train_df[CLASSIFICATION_TARGET])
    classical_scores = text_pipeline.predict_proba(test_df[TEXT_FEATURE])[:, 1]
    classical_result = classification_metrics(y_test, classical_scores, threshold=0.5)

    # LLM-style zero-shot proxy: keyword heuristic (no trained model, no probability scores)
    heuristic_preds = test_df[TEXT_FEATURE].map(_heuristic_llm_style_label).astype(int)
    heuristic_result = {
        "accuracy": round(float(accuracy_score(y_test, heuristic_preds)), 4),
        "precision": round(float(precision_score(y_test, heuristic_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, heuristic_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, heuristic_preds, zero_division=0)), 4),
    }

    write_json(
        output_dir / "llm_vs_classical_benchmark.json",
        {
            "comparison_type": "classical_text_model_vs_llm_style_zero_shot_proxy",
            "classical_metrics": classical_result,
            "llm_style_proxy_metrics": heuristic_result,
        },
    )
