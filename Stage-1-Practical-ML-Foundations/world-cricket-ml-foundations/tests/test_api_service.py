from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from world_cricket_ml.serving.prediction_service import PredictionService
from world_cricket_ml.utils import ensure_dir, write_json


class PredictConstant:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


class PredictProbaConstant:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame):
        return np.column_stack((
            np.full(len(frame), 1 - self.probability),
            np.full(len(frame), self.probability),
        ))


@pytest.fixture()
def built_project(tmp_path: Path) -> Path:
    project_root = tmp_path
    ensure_dir(project_root / "artifacts" / "classification")
    ensure_dir(project_root / "artifacts" / "regression")
    ensure_dir(project_root / "data" / "raw")

    frame = pd.DataFrame([
        {
            "team": "India",
            "opponent": "Australia",
            "venue": "Wankhede",
            "city": "Mumbai",
            "tournament": "Series",
            "toss_decision": "bat",
            "season": "2025",
            "won_toss": 1,
            "batted_first": 1,
            "team_runs": 175,
            "team_wickets_lost": 6,
            "opponent_runs": 162,
            "opponent_wickets_lost": 8,
            "team_run_rate": 8.75,
            "opponent_run_rate": 8.1,
            "match_nrr": 0.65,
            "rolling_3_win_rate": 0.67,
            "rolling_5_win_rate": 0.8,
            "rolling_10_win_rate": 0.7,
            "rolling_3_avg_nrr": 0.3,
            "rolling_5_avg_nrr": 0.45,
            "rolling_10_avg_nrr": 0.4,
            "rolling_3_avg_runs": 168.0,
            "rolling_5_avg_runs": 170.0,
            "rolling_10_avg_runs": 166.0,
            "matches_played_so_far": 25,
            "rest_days": 6.0,
            "opponent_rolling_5_win_rate": 0.55,
            "year": 2025,
            "month": 12,
            "quarter": 4,
        },
        {
            "team": "Sri Lanka",
            "opponent": "Pakistan",
            "venue": "Colombo",
            "city": "Colombo",
            "tournament": "Series",
            "toss_decision": "field",
            "season": "2025",
            "won_toss": 0,
            "batted_first": 0,
            "team_runs": 168,
            "team_wickets_lost": 7,
            "opponent_runs": 166,
            "opponent_wickets_lost": 9,
            "team_run_rate": 8.4,
            "opponent_run_rate": 8.3,
            "match_nrr": 0.1,
            "rolling_3_win_rate": 0.5,
            "rolling_5_win_rate": 0.6,
            "rolling_10_win_rate": 0.55,
            "rolling_3_avg_nrr": 0.12,
            "rolling_5_avg_nrr": 0.18,
            "rolling_10_avg_nrr": 0.09,
            "rolling_3_avg_runs": 161.0,
            "rolling_5_avg_runs": 164.0,
            "rolling_10_avg_runs": 160.0,
            "matches_played_so_far": 22,
            "rest_days": 5.0,
            "opponent_rolling_5_win_rate": 0.52,
            "year": 2025,
            "month": 11,
            "quarter": 4,
        },
    ])
    frame.to_csv(project_root / "data" / "raw" / "world_cricket_latest_snapshots.csv", index=False)

    joblib.dump(PredictProbaConstant(1.0), project_root / "artifacts" / "classification" / "model.joblib")
    joblib.dump(PredictProbaConstant(0.25), project_root / "artifacts" / "classification" / "calibrated_model.joblib")
    joblib.dump(PredictConstant(0.73), project_root / "artifacts" / "regression" / "model.joblib")
    write_json(
        project_root / "artifacts" / "classification" / "metrics.json",
        {
            "best_model": "dummy",
            "candidate_metrics": {"dummy": {"roc_auc": 0.9, "threshold": 0.5}},
            "calibrated_model_metrics": {"roc_auc": 0.7},
            "selected_serving_artifact": "raw_model",
            "serving_threshold": 0.5,
        },
    )
    return project_root


def test_prediction_service_requires_artifacts() -> None:
    with pytest.raises(FileNotFoundError):
        PredictionService(Path("C:/missing-project"))


def test_prediction_service_uses_selected_artifact(built_project: Path) -> None:
    service = PredictionService(built_project)
    assert service.selected_classifier_artifact == "raw_model"
    assert service.health_status()["teams_loaded"] == 2
    prediction = service.predict_team("india")
    assert prediction["team"] == "India"
    assert prediction["dominance_probability"] == 1.0
    assert prediction["predicted_future_win_rate"] == 0.73


@pytest.mark.parametrize("alias", ["India", "india", "INDIA", " ind ia "])
def test_predict_team_is_case_and_spacing_insensitive(built_project: Path, alias: str) -> None:
    service = PredictionService(built_project)
    prediction = service.predict_team(alias)
    assert prediction["team"] == "India"


def test_health_status_includes_data_loaded_at(built_project: Path) -> None:
    service = PredictionService(built_project)
    status = service.health_status()
    assert status["status"] == "ok"
    assert "data_loaded_at" in status
    assert status["data_loaded_at"]  # non-empty ISO timestamp


def test_predict_team_aliases_map_to_canonical_team_name(built_project: Path) -> None:
    service = PredictionService(built_project)
    prediction = service.predict_team("Srilanka")
    assert prediction["team"] == "Sri Lanka"



def test_predict_team_unknown_raises_key_error(built_project: Path) -> None:
    service = PredictionService(built_project)
    with pytest.raises(KeyError):
        service.predict_team("UnknownTeamXYZ")


def test_leaderboard_returns_all_teams(built_project: Path) -> None:
    service = PredictionService(built_project)
    board = service.leaderboard()
    assert len(board) == 2
    assert {item["team"] for item in board} == {"India", "Sri Lanka"}
    assert "downfall_risk" in board[0]
    assert "surprise_score" in board[0]
