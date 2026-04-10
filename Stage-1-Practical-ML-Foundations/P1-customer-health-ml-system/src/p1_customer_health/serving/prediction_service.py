from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import pandas as pd

from p1_customer_health.app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class PredictionService:
    classifier_bundle: dict
    regressor_bundle: dict
    segmenter_bundle: dict
    anomaly_bundle: dict

    @classmethod
    def from_settings(cls, settings: Settings) -> "PredictionService":
        logger.info("Loading classifier artifact from %s", settings.classifier_artifact)
        classifier_bundle = joblib.load(settings.classifier_artifact)
        logger.info("Loading regressor artifact from %s", settings.regressor_artifact)
        regressor_bundle = joblib.load(settings.regressor_artifact)
        logger.info("Loading segmenter artifact from %s", settings.segmenter_artifact)
        segmenter_bundle = joblib.load(settings.segmenter_artifact)
        logger.info("Loading anomaly detector artifact from %s", settings.anomaly_artifact)
        anomaly_bundle = joblib.load(settings.anomaly_artifact)
        logger.info(
            "All artifacts loaded. Classifier trained_at=%s",
            classifier_bundle.get("trained_at", "unknown"),
        )
        return cls(
            classifier_bundle=classifier_bundle,
            regressor_bundle=regressor_bundle,
            segmenter_bundle=segmenter_bundle,
            anomaly_bundle=anomaly_bundle,
        )

    @property
    def model_version(self) -> str:
        trained_at = self.classifier_bundle.get("trained_at")
        if trained_at:
            return f"p1_customer_health_{trained_at}"
        return "p1_customer_health_v1"

    def predict(self, records: list[dict]) -> list[dict]:
        df = pd.DataFrame(records)
        classifier = self.classifier_bundle["model"]
        threshold = float(self.classifier_bundle["threshold"])
        regressor = self.regressor_bundle["model"]
        segmenter = self.segmenter_bundle["model"]
        segmenter_preprocessor = self.segmenter_bundle["preprocessor"]
        anomaly = self.anomaly_bundle["model"]
        anomaly_preprocessor = self.anomaly_bundle["preprocessor"]

        churn_scores = classifier.predict_proba(df)[:, 1]
        revenue_preds = regressor.predict(df)
        segment_features = segmenter_preprocessor.transform(df)
        segment_ids = segmenter.predict(segment_features)
        anomaly_flags = (anomaly.predict(anomaly_preprocessor.transform(df)) == -1).astype(int)

        logger.info("Predicted %d records (churn threshold=%.3f)", len(df), threshold)

        return [
            {
                "account_id": account_id,
                "churn_score": round(float(churn_scores[idx]), 4),
                "churn_label": int(churn_scores[idx] >= threshold),
                "predicted_revenue_change": round(float(revenue_preds[idx]), 2),
                "segment_id": int(segment_ids[idx]),
                "anomaly_flag": int(anomaly_flags[idx]),
            }
            for idx, account_id in enumerate(df["account_id"].tolist())
        ]
