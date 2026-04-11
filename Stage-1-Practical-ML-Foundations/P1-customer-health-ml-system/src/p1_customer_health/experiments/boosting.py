from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from p1_customer_health.domain.dataset import CLASSIFICATION_TARGET, time_split
from p1_customer_health.training.metrics import best_threshold, classification_metrics, ensure_dir, write_json
from p1_customer_health.training.preprocessing import dense_tabular_preprocessor


def run_boosting_benchmark(df: pd.DataFrame, artifact_root: Path) -> None:
    output_dir = artifact_root / "boosting"
    ensure_dir(output_dir)
    split = time_split(df)
    train_df = split.train
    val_df = split.validation
    test_df = split.test
    y_train = train_df[CLASSIFICATION_TARGET]
    y_val = val_df[CLASSIFICATION_TARGET]
    y_test = test_df[CLASSIFICATION_TARGET]
    preprocessor = dense_tabular_preprocessor()
    x_train = preprocessor.fit_transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)

    candidates: list[tuple[str, Any]] = []
    try:
        from xgboost import XGBClassifier
        candidates.append(("xgboost", XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)))
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        candidates.append(("lightgbm", LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)))
    except ImportError:
        pass

    if not candidates:
        write_json(output_dir / "status.json", {"status": "skipped", "reason": "xgboost and lightgbm not installed"})
        return

    leaderboard: list[dict[str, Any]] = []
    for name, model in candidates:
        model.fit(x_train, y_train)
        # Tune threshold on validation set; evaluate on held-out test set
        val_scores = model.predict_proba(x_val)[:, 1]
        threshold = best_threshold(y_val, val_scores)
        test_scores = model.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test, test_scores, threshold)
        leaderboard.append({"model": name, "threshold": threshold, "test_metrics": metrics})
    write_json(output_dir / "status.json", {"status": "completed", "leaderboard": leaderboard})
