from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

# FEATURE_COLUMNS comes from preprocessing (domain knowledge), NOT from training.
# Serving must never depend on the training module.
from world_cricket_ml.training.preprocessing import FEATURE_COLUMNS
from world_cricket_ml.utils import compute_team_signals, read_json

log = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        log.info("Loading PredictionService from %s", project_root)

        self.classification_metrics = read_json(
            project_root / "artifacts" / "classification" / "metrics.json"
        )
        selected_artifact = self.classification_metrics.get("selected_serving_artifact", "calibrated_model")
        model_name = "calibrated_model.joblib" if selected_artifact == "calibrated_model" else "model.joblib"
        self.selected_classifier_artifact = selected_artifact
        self.classifier = joblib.load(project_root / "artifacts" / "classification" / model_name)
        self.regressor = joblib.load(project_root / "artifacts" / "regression" / "model.joblib")
        self.latest = pd.read_csv(project_root / "data" / "raw" / "world_cricket_latest_snapshots.csv")
        self._data_loaded_at = datetime.now(tz=timezone.utc).isoformat()
        log.info(
            "Service ready — artifact=%s  teams=%d",
            selected_artifact,
            len(self.latest),
        )

    def health_status(self) -> dict:
        return {
            "status": "ok",
            "teams_loaded": int(len(self.latest)),
            "selected_classifier_artifact": self.selected_classifier_artifact,
            "data_loaded_at": self._data_loaded_at,
        }

    def leaderboard(self) -> list[dict]:
        frame = self.latest.copy()
        frame["dominance_probability"] = self.classifier.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        frame["predicted_future_win_rate"] = self.regressor.predict(frame[FEATURE_COLUMNS])
        frame = compute_team_signals(frame)
        frame = frame.sort_values("dominance_probability", ascending=False)
        return frame[[
            "team",
            "dominance_probability",
            "predicted_future_win_rate",
            "downfall_risk",
            "surprise_score",
            "rolling_5_win_rate",
            "rolling_5_avg_nrr",
        ]].to_dict(orient="records")

    def predict_team(self, team: str) -> dict:
        frame = self.latest[self.latest["team"].str.lower() == team.lower()].copy()
        if frame.empty:
            raise KeyError(team)
        row = frame.iloc[[0]].copy()
        row["dominance_probability"] = self.classifier.predict_proba(row[FEATURE_COLUMNS])[:, 1]
        row["predicted_future_win_rate"] = self.regressor.predict(row[FEATURE_COLUMNS])
        row = compute_team_signals(row)

        return {
            "team": row["team"].iloc[0],
            "dominance_probability": round(float(row["dominance_probability"].iloc[0]), 4),
            "predicted_future_win_rate": round(float(row["predicted_future_win_rate"].iloc[0]), 4),
            "downfall_risk": round(float(row["downfall_risk"].iloc[0]), 4),
            "surprise_score": round(float(row["surprise_score"].iloc[0]), 4),
            "recent_form": {
                "rolling_5_win_rate": round(float(row["rolling_5_win_rate"].iloc[0]), 4),
                "rolling_5_avg_nrr": round(float(row["rolling_5_avg_nrr"].iloc[0]), 4),
            },
        }
