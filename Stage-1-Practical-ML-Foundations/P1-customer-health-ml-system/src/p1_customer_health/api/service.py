from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from p1_customer_health.config import Settings


@dataclass
class PredictionService:
    classifier_bundle: dict
    regressor_bundle: dict
    segmenter_bundle: dict
    anomaly_bundle: dict

    @classmethod
    def from_settings(cls, settings: Settings) -> "PredictionService":
        return cls(
            classifier_bundle=joblib.load(settings.classifier_artifact),
            regressor_bundle=joblib.load(settings.regressor_artifact),
            segmenter_bundle=joblib.load(settings.segmenter_artifact),
            anomaly_bundle=joblib.load(settings.anomaly_artifact),
        )

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

        results = []
        for idx, account_id in enumerate(df["account_id"].tolist()):
            results.append(
                {
                    "account_id": account_id,
                    "churn_score": round(float(churn_scores[idx]), 4),
                    "churn_label": int(churn_scores[idx] >= threshold),
                    "predicted_revenue_change": round(float(revenue_preds[idx]), 2),
                    "segment_id": int(segment_ids[idx]),
                    "anomaly_flag": int(anomaly_flags[idx]),
                }
            )
        return results
