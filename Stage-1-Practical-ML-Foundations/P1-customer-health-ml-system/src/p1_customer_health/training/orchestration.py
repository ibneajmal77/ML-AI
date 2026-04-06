from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p1_customer_health.analysis.audit import write_data_quality_report
from p1_customer_health.analysis.business_framing import write_business_decision_workflow
from p1_customer_health.analysis.leakage import write_leakage_report
from p1_customer_health.domain.dataset import CLASSIFICATION_TARGET, NUMERIC_FEATURES, load_dataset, time_split
from p1_customer_health.experiments.boosting import run_boosting_benchmark
from p1_customer_health.experiments.llm_benchmark import run_llm_vs_classical_benchmark
from p1_customer_health.experiments.onnx_export import export_classifier_to_onnx, export_dense_classifier_to_onnx
from p1_customer_health.experiments.rl import run_contextual_bandit
from p1_customer_health.experiments.self_supervised import run_self_supervised_benchmark
from p1_customer_health.training.classification import train_classifier
from p1_customer_health.training.metrics import ensure_dir
from p1_customer_health.training.regression import train_regressor
from p1_customer_health.training.unsupervised import train_unsupervised


def export_verified_onnx_path(df, artifact_root: Path) -> None:
    output_dir = artifact_root / "onnx"
    ensure_dir(output_dir)
    split = time_split(df)
    train_df = split.train
    test_df = split.test
    x_train = train_df[NUMERIC_FEATURES].to_numpy(dtype="float32")
    x_test = test_df[NUMERIC_FEATURES].to_numpy(dtype="float32")
    y_train = train_df[CLASSIFICATION_TARGET]
    dense_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    dense_model.fit(x_train, y_train)
    joblib.dump({"model": dense_model, "task": "onnx_dense_classifier"}, output_dir / "dense_model.joblib")
    export_dense_classifier_to_onnx(dense_model, x_test, output_dir)


def train_all(dataset_path: Path, artifact_root: Path) -> None:
    df = load_dataset(dataset_path)
    ensure_dir(artifact_root)
    write_business_decision_workflow(artifact_root / "business")
    write_data_quality_report(df, artifact_root / "data_quality")
    split = time_split(df)
    write_leakage_report(split.train, split.validation, split.test, artifact_root / "leakage")
    train_classifier(df, artifact_root)
    train_regressor(df, artifact_root)
    train_unsupervised(df, artifact_root)
    run_self_supervised_benchmark(df, artifact_root)
    run_boosting_benchmark(df, artifact_root)
    run_llm_vs_classical_benchmark(df, artifact_root / "llm_benchmark")
    run_contextual_bandit(df, artifact_root / "reinforcement_learning")
    classifier_bundle = joblib.load(artifact_root / "classification" / "model.joblib")
    export_classifier_to_onnx(classifier_bundle, split.test.head(5), artifact_root / "onnx")
    export_verified_onnx_path(df, artifact_root)
