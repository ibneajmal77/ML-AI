from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from p1_customer_health.domain.dataset import CLASSIFICATION_TARGET, TEXT_FEATURE, time_split
from p1_customer_health.training.metrics import classification_metrics, ensure_dir, write_json


def run_self_supervised_benchmark(df: pd.DataFrame, artifact_root: Path) -> None:
    output_dir = artifact_root / "self_supervised"
    ensure_dir(output_dir)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None

    split = time_split(df)
    if SentenceTransformer is not None:
        try:
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            train_embeddings = encoder.encode(split.train[TEXT_FEATURE].tolist())
            test_embeddings = encoder.encode(split.test[TEXT_FEATURE].tolist())
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(train_embeddings, split.train[CLASSIFICATION_TARGET])
            scores = model.predict_proba(test_embeddings)[:, 1]
            metrics = classification_metrics(split.test[CLASSIFICATION_TARGET], scores, threshold=0.5)
            write_json(output_dir / "status.json", {"status": "completed", "method": "sentence_transformers", "test_metrics": metrics})
            return
        except Exception:
            pass

    vectorizer = TfidfVectorizer(max_features=600, ngram_range=(1, 2))
    train_sparse = vectorizer.fit_transform(split.train[TEXT_FEATURE])
    test_sparse = vectorizer.transform(split.test[TEXT_FEATURE])
    reducer = TruncatedSVD(n_components=32, random_state=42)
    train_embeddings = reducer.fit_transform(train_sparse)
    test_embeddings = reducer.transform(test_sparse)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train_embeddings, split.train[CLASSIFICATION_TARGET])
    scores = model.predict_proba(test_embeddings)[:, 1]
    metrics = classification_metrics(split.test[CLASSIFICATION_TARGET], scores, threshold=0.5)
    write_json(output_dir / "status.json", {"status": "completed", "method": "local_dense_text_embedding_fallback", "test_metrics": metrics})
