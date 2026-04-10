import math

import pandas as pd
import pytest

from world_cricket_ml.analysis.leakage import build_leakage_report
from world_cricket_ml.app.settings import rolling_two_year_quarterly_window
from world_cricket_ml.domain.dataset import build_snapshot_frame
from world_cricket_ml.training.classification import FEATURE_COLUMNS, time_split
from world_cricket_ml.training.metrics import classification_metrics, regression_metrics
from world_cricket_ml.training.splits import three_way_time_split


# ---------------------------------------------------------------------------
# Dataset / snapshot tests
# ---------------------------------------------------------------------------

def test_snapshot_frame_builds_labels() -> None:
    rows = []
    for idx in range(12):
        rows.append({
            "match_id": f"m{idx}",
            "match_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx * 5),
            "season": "2024",
            "team": "India",
            "opponent": "Australia",
            "venue": "Some Stadium",
            "city": "Some City",
            "tournament": "Series",
            "toss_winner": "India",
            "toss_decision": "bat",
            "won_toss": 1,
            "batted_first": 1,
            "won_match": 1 if idx % 2 == 0 else 0,
            "lost_match": 0 if idx % 2 == 0 else 1,
            "is_no_result": 0,
            "margin_runs": 10,
            "margin_wickets": 0,
            "team_runs": 160 + idx,
            "team_wickets_lost": 5,
            "team_overs": 20.0,
            "opponent_runs": 150,
            "opponent_wickets_lost": 8,
            "opponent_overs": 20.0,
            "team_run_rate": 8.0,
            "opponent_run_rate": 7.5,
            "match_nrr": 0.5,
            "power_index_hint": 180,
            "match_text": "India beat Australia comfortably.",
        })
    frame = build_snapshot_frame(pd.DataFrame(rows))
    assert "dominant_next_cycle" in frame.columns
    assert "rolling_5_win_rate" in frame.columns
    assert frame["rolling_5_win_rate"].iloc[0] == 0.0
    assert frame["future_win_rate_next_5"].between(0.0, 1.0).all()


def test_time_split_is_chronological() -> None:
    frame = pd.DataFrame(
        {
            "match_date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "future_matches_available": [3] * 10,
        }
    )
    train_df, test_df = time_split(frame)
    assert not train_df.empty
    assert not test_df.empty
    assert train_df["match_date"].max() < test_df["match_date"].min()


def test_rolling_two_year_quarterly_window_uses_completed_quarters() -> None:
    start_date, end_date = rolling_two_year_quarterly_window(reference=pd.Timestamp("2026-04-07").date())
    assert str(start_date) == "2024-04-01"
    assert str(end_date) == "2026-03-31"


def test_leakage_report_matches_feature_contract(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "future_win_rate_next_5": [0.5],
            "dominant_next_cycle": [1],
            "rolling_5_win_rate": [0.7],
        }
    )
    report = build_leakage_report(frame, FEATURE_COLUMNS, tmp_path)
    assert report["status"] == "ok"
    assert "future_win_rate_next_5" in report["blocked_from_training"]
    assert report["implemented_holdout"] == "single time-based holdout split"


def test_leakage_report_includes_mi_scan(tmp_path) -> None:
    """Mutual information scan should be present in the leakage report."""
    frame = pd.DataFrame(
        {
            "future_win_rate_next_5": [0.5] * 30,
            "dominant_next_cycle": ([1] * 15) + ([0] * 15),
            "rolling_5_win_rate": [float(i) / 30 for i in range(30)],
        }
    )
    report = build_leakage_report(frame, FEATURE_COLUMNS, tmp_path)
    assert "mutual_info_scan" in report
    assert "high_mi_features" in report
    assert isinstance(report["mutual_info_scan"], dict)
    assert isinstance(report["high_mi_features"], list)


# ---------------------------------------------------------------------------
# Three-way split tests
# ---------------------------------------------------------------------------

def test_three_way_time_split_non_overlapping() -> None:
    frame = pd.DataFrame({"match_date": pd.date_range("2024-01-01", periods=30, freq="D")})
    train, val, test = three_way_time_split(frame, train_frac=0.6, val_frac=0.2)
    assert not train.empty
    assert not val.empty
    assert not test.empty
    # Chronological ordering: train < val < test
    assert train["match_date"].max() < val["match_date"].min()
    assert val["match_date"].max() < test["match_date"].min()


def test_three_way_time_split_covers_all_rows() -> None:
    frame = pd.DataFrame({"match_date": pd.date_range("2024-01-01", periods=30, freq="D")})
    train, val, test = three_way_time_split(frame)
    assert len(train) + len(val) + len(test) == len(frame)


def test_three_way_time_split_invalid_fracs() -> None:
    frame = pd.DataFrame({"match_date": pd.date_range("2024-01-01", periods=10, freq="D")})
    with pytest.raises(ValueError):
        three_way_time_split(frame, train_frac=0.7, val_frac=0.4)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_regression_metrics_includes_mape() -> None:
    y_true = [0.5, 0.6, 0.7, 0.8]
    y_pred = [0.45, 0.65, 0.72, 0.78]
    metrics = regression_metrics(y_true, y_pred)
    assert "mape" in metrics
    assert metrics["mape"] is not None
    assert metrics["mape"] >= 0.0
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_regression_metrics_mape_excludes_zero_actuals() -> None:
    """MAPE should not be NaN when some actual values are zero."""
    y_true = [0.0, 0.5, 0.6]
    y_pred = [0.1, 0.5, 0.6]
    metrics = regression_metrics(y_true, y_pred)
    # MAPE should be computed only on non-zero actuals (0.5 and 0.6) → 0
    assert metrics["mape"] is not None
    assert not math.isnan(metrics["mape"])


def test_regression_metrics_all_zero_actuals() -> None:
    """When all actuals are zero MAPE should be None (undefined)."""
    y_true = [0.0, 0.0, 0.0]
    y_pred = [0.1, 0.2, 0.3]
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mape"] is None


def test_classification_metrics_threshold_sweep() -> None:
    """Widening threshold from 0 to 1 should move precision up and recall down."""
    import numpy as np
    y_true = [1, 1, 0, 0, 1]
    probs = [0.8, 0.7, 0.3, 0.2, 0.6]
    low = classification_metrics(y_true, probs, threshold=0.1)
    high = classification_metrics(y_true, probs, threshold=0.9)
    # At very low threshold everything is predicted positive → high recall, low precision
    assert low["recall"] >= high["recall"]
    # At very high threshold nothing is predicted positive → lower recall
    assert high["recall"] <= low["recall"]
