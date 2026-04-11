from pathlib import Path

import joblib
import pytest

from p1_customer_health.domain.synthetic_data import generate_customer_health_data
from p1_customer_health.serving.prediction_service import PredictionService
from p1_customer_health.training.orchestration import train_all


@pytest.fixture(scope="module")
def trained_service(tmp_path_factory):
    """Train on a small dataset once and return a loaded PredictionService."""
    tmp = tmp_path_factory.mktemp("artifacts")
    data_path = tmp / "customer_health.csv"
    artifact_root = tmp / "artifacts"
    df = generate_customer_health_data(n_samples=300, seed=7)
    df.to_csv(data_path, index=False)
    train_all(data_path, artifact_root)
    return PredictionService(
        classifier_bundle=joblib.load(artifact_root / "classification" / "model.joblib"),
        regressor_bundle=joblib.load(artifact_root / "regression" / "model.joblib"),
        segmenter_bundle=joblib.load(artifact_root / "unsupervised" / "segmenter.joblib"),
        anomaly_bundle=joblib.load(artifact_root / "unsupervised" / "anomaly_detector.joblib"),
    )


def test_prediction_returns_correct_number_of_rows(trained_service):
    df = generate_customer_health_data(n_samples=10, seed=1)
    records = df.drop(columns=["churned_30d", "revenue_change_next_30d"]).to_dict(orient="records")
    predictions = trained_service.predict(records)
    assert len(predictions) == 10


def test_prediction_output_fields(trained_service):
    df = generate_customer_health_data(n_samples=5, seed=2)
    records = df.drop(columns=["churned_30d", "revenue_change_next_30d"]).to_dict(orient="records")
    result = trained_service.predict(records)
    for row in result:
        assert "account_id" in row
        assert "churn_score" in row
        assert "churn_label" in row
        assert "predicted_revenue_change" in row
        assert "segment_id" in row
        assert "anomaly_flag" in row


def test_churn_score_is_probability(trained_service):
    df = generate_customer_health_data(n_samples=50, seed=3)
    records = df.drop(columns=["churned_30d", "revenue_change_next_30d"]).to_dict(orient="records")
    predictions = trained_service.predict(records)
    for row in predictions:
        assert 0.0 <= row["churn_score"] <= 1.0, f"churn_score out of range: {row['churn_score']}"
        assert row["churn_label"] in (0, 1)
        assert row["anomaly_flag"] in (0, 1)


def test_model_version_is_set(trained_service):
    # trained_at is embedded in the bundle at training time
    assert trained_service.model_version != ""
    assert "p1_customer_health" in trained_service.model_version
