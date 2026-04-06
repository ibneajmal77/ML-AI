from pathlib import Path

import joblib

from p1_customer_health.domain.synthetic_data import generate_customer_health_data
from p1_customer_health.serving.prediction_service import PredictionService
from p1_customer_health.training.orchestration import train_all


def test_prediction_service_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "customer_health.csv"
    artifact_root = tmp_path / "artifacts"
    df = generate_customer_health_data(n_samples=250, seed=7)
    df.to_csv(data_path, index=False)

    train_all(data_path, artifact_root)

    service = PredictionService(
        classifier_bundle=joblib.load(artifact_root / "classification" / "model.joblib"),
        regressor_bundle=joblib.load(artifact_root / "regression" / "model.joblib"),
        segmenter_bundle=joblib.load(artifact_root / "unsupervised" / "segmenter.joblib"),
        anomaly_bundle=joblib.load(artifact_root / "unsupervised" / "anomaly_detector.joblib"),
    )

    records = df.head(3).drop(columns=["churned_30d", "revenue_change_next_30d"]).to_dict(orient="records")
    predictions = service.predict(records)

    assert len(predictions) == 3
    assert "churn_score" in predictions[0]
    assert "predicted_revenue_change" in predictions[0]
