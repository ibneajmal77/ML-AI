from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from world_cricket_ml.app.settings import (
    REFRESH_CADENCE_MONTHS,
    TOP_20_T20I_TEAMS,
    rolling_two_year_quarterly_window,
)
from world_cricket_ml.utils import ensure_dir, write_json

CRICSHEET_ZIP_URL = "https://cricsheet.org/downloads/all_json.zip"


@dataclass(frozen=True)
class DatasetBuildResult:
    raw_matches_path: Path
    snapshots_path: Path
    latest_path: Path
    metadata_path: Path


def _team_alias(team_name: str) -> str:
    aliases = {
        "United States of America": "USA",
        "U.S.A.": "USA",
        "U.A.E.": "United Arab Emirates",
    }
    return aliases.get(team_name, team_name)


def _download_cricsheet_zip(target: Path) -> None:
    ensure_dir(target.parent)
    with urllib.request.urlopen(CRICSHEET_ZIP_URL, timeout=120) as response:
        target.write_bytes(response.read())


def _extract_runs_and_wickets(innings: dict) -> tuple[int, int, float]:
    deliveries = 0
    runs = 0
    wickets = 0
    for over in innings.get("overs", []):
        for delivery in over.get("deliveries", []):
            deliveries += 1
            runs += int(delivery.get("runs", {}).get("total", 0))
            wickets += len(delivery.get("wickets", []))
    overs = round(deliveries / 6.0, 2) if deliveries else 0.0
    return runs, wickets, overs


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _team_rows_from_match(match_id: str, payload: dict, selected_teams: set[str]) -> list[dict]:
    info = payload.get("info", {})
    if info.get("gender") != "male":
        return []
    if info.get("match_type") not in {"T20", "T20I"}:
        return []
    if info.get("team_type") != "international":
        return []

    teams = [_team_alias(team) for team in info.get("teams", [])]
    if len(teams) != 2 or not set(teams).issubset(selected_teams):
        return []

    match_date = pd.to_datetime(info.get("dates", [None])[0])
    start_date, end_date = rolling_two_year_quarterly_window()
    if pd.isna(match_date) or not (pd.Timestamp(start_date) <= match_date <= pd.Timestamp(end_date)):
        return []

    innings_by_team: dict[str, tuple[int, int, float]] = {}
    innings_list = payload.get("innings", [])
    for innings in innings_list:
        innings_team = _team_alias(innings.get("team", ""))
        innings_by_team[innings_team] = _extract_runs_and_wickets(innings)
    if len(innings_by_team) < 2:
        return []

    outcome = info.get("outcome", {})
    winner = _team_alias(outcome.get("winner", ""))
    toss = info.get("toss", {})
    event = info.get("event", {})
    venue = info.get("venue", "Unknown")
    city = info.get("city", "Unknown")
    tournament = event.get("name", "Bilateral / Other")
    by_runs = int(outcome.get("by", {}).get("runs", 0))
    by_wickets = int(outcome.get("by", {}).get("wickets", 0))
    no_result = int(outcome.get("result") in {"no result", "tie"})

    rows: list[dict] = []
    first_innings_team = _team_alias(innings_list[0].get("team", "")) if innings_list else ""
    for team in teams:
        opponent = next(item for item in teams if item != team)
        team_runs, team_wickets, team_overs = innings_by_team.get(team, (0, 0, 0.0))
        opp_runs, opp_wickets, opp_overs = innings_by_team.get(opponent, (0, 0, 0.0))
        team_batted_first = int(first_innings_team == team)
        team_run_rate = _safe_div(team_runs, max(team_overs, 1.0))
        opp_run_rate = _safe_div(opp_runs, max(opp_overs, 1.0))
        match_nrr = round(team_run_rate - opp_run_rate, 4)
        rows.append(
            {
                "match_id": match_id.removesuffix(".json"),
                "match_date": match_date.normalize(),
                "season": str(info.get("season", match_date.year)),
                "team": team,
                "opponent": opponent,
                "venue": venue,
                "city": city,
                "tournament": tournament,
                "toss_winner": _team_alias(toss.get("winner", "Unknown")),
                "toss_decision": toss.get("decision", "unknown"),
                "won_toss": int(_team_alias(toss.get("winner", "")) == team),
                "batted_first": team_batted_first,
                "won_match": int(winner == team),
                "lost_match": int(winner == opponent),
                "is_no_result": no_result,
                "margin_runs": by_runs if winner == team else -by_runs if by_runs else 0,
                "margin_wickets": by_wickets if winner == team else -by_wickets if by_wickets else 0,
                "team_runs": team_runs,
                "team_wickets_lost": team_wickets,
                "team_overs": team_overs,
                "opponent_runs": opp_runs,
                "opponent_wickets_lost": opp_wickets,
                "opponent_overs": opp_overs,
                "team_run_rate": round(team_run_rate, 4),
                "opponent_run_rate": round(opp_run_rate, 4),
                "match_nrr": match_nrr,
                "power_index_hint": round(team_runs + (10 - team_wickets) * 3, 2),
                "match_text": (
                    f"{team} vs {opponent} in {tournament} at {venue}. "
                    f"{team} scored {team_runs} for {team_wickets} in {team_overs} overs. "
                    f"Net run rate impact was {match_nrr:.2f}."
                ),
            }
        )
    return rows


def build_match_dataframe(raw_zip_path: Path, selected_teams: Iterable[str] = TOP_20_T20I_TEAMS) -> pd.DataFrame:
    selected_team_set = {_team_alias(team) for team in selected_teams}
    rows: list[dict] = []
    with zipfile.ZipFile(raw_zip_path) as archive:
        for match_name in archive.namelist():
            if not match_name.endswith('.json'):
                continue
            payload = json.loads(archive.read(match_name))
            rows.extend(_team_rows_from_match(match_name, payload, selected_team_set))
    matches = pd.DataFrame(rows).sort_values(["match_date", "team", "match_id"]).reset_index(drop=True)
    return matches


def build_snapshot_frame(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)
    group = df.groupby("team", group_keys=False)

    for window in (3, 5, 10):
        df[f"rolling_{window}_win_rate"] = group["won_match"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        ).fillna(0.0)
        df[f"rolling_{window}_avg_nrr"] = group["match_nrr"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        ).fillna(0.0)
        df[f"rolling_{window}_avg_runs"] = group["team_runs"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        ).fillna(0.0)

    df["matches_played_so_far"] = group.cumcount()
    df["rest_days"] = group["match_date"].diff().dt.days.fillna(7).clip(lower=0).astype(float)
    df["year"] = df["match_date"].dt.year
    df["month"] = df["match_date"].dt.month
    df["quarter"] = df["match_date"].dt.quarter

    opponent_recent_form = df[["team", "match_date", "rolling_5_win_rate"]].rename(
        columns={"team": "opponent", "rolling_5_win_rate": "opponent_rolling_5_win_rate"}
    )
    df = df.merge(opponent_recent_form, how="left", on=["opponent", "match_date"])
    df["opponent_rolling_5_win_rate"] = df["opponent_rolling_5_win_rate"].fillna(0.0)

    future_group = df.groupby("team", group_keys=False)
    future_window = 5
    df["future_matches_available"] = future_group["won_match"].transform(
        lambda s: s.shift(-1).rolling(future_window, min_periods=1).count()
    )
    df["future_win_rate_next_5"] = future_group["won_match"].transform(
        lambda s: s[::-1].shift(1).rolling(future_window, min_periods=1).mean()[::-1]
    )
    df["future_avg_nrr_next_5"] = future_group["match_nrr"].transform(
        lambda s: s[::-1].shift(1).rolling(future_window, min_periods=1).mean()[::-1]
    )
    df["future_win_rate_next_5"] = df["future_win_rate_next_5"].fillna(df["rolling_5_win_rate"])
    df["future_avg_nrr_next_5"] = df["future_avg_nrr_next_5"].fillna(df["rolling_5_avg_nrr"])
    df["recent_to_future_delta"] = df["future_win_rate_next_5"] - df["rolling_5_win_rate"]
    df["dominant_next_cycle"] = (
        (df["future_matches_available"] >= 3)
        & (df["future_win_rate_next_5"] >= 0.7)
        & (df["future_avg_nrr_next_5"] >= 0.35)
    ).astype(int)
    df["downfall_next_cycle"] = (
        (df["future_matches_available"] >= 3)
        & (df["rolling_5_win_rate"] >= 0.55)
        & (df["future_win_rate_next_5"] <= 0.4)
    ).astype(int)
    df["surprise_candidate"] = (
        (df["future_matches_available"] >= 3)
        & (df["rolling_10_win_rate"] <= 0.5)
        & (df["future_win_rate_next_5"] >= 0.6)
    ).astype(int)
    df["outlook_label"] = np.select(
        [df["dominant_next_cycle"] == 1, df["downfall_next_cycle"] == 1, df["surprise_candidate"] == 1],
        ["dominate", "downfall", "surprise"],
        default="stable",
    )

    numeric_fill = [col for col in df.columns if df[col].dtype.kind in {"f", "i"}]
    df[numeric_fill] = df[numeric_fill].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def build_project_datasets(project_root: Path) -> DatasetBuildResult:
    data_root = ensure_dir(project_root / "data")
    data_raw = ensure_dir(data_root / "raw")
    zip_path = data_raw / "cricsheet_all_json.zip"
    raw_matches_path = data_raw / "world_cricket_team_matches.csv"
    snapshots_path = data_raw / "world_cricket_snapshots.csv"
    latest_path = data_raw / "world_cricket_latest_snapshots.csv"
    metadata_path = data_raw / "dataset_metadata.json"
    obsolete_prepared_dir = data_root / "prepared"

    if obsolete_prepared_dir.exists():
        shutil.rmtree(obsolete_prepared_dir)

    if not zip_path.exists():
        _download_cricsheet_zip(zip_path)

    matches = build_match_dataframe(zip_path)
    snapshots = build_snapshot_frame(matches)
    latest = snapshots.sort_values("match_date").groupby("team").tail(1).reset_index(drop=True)

    matches.to_csv(raw_matches_path, index=False)
    snapshots.to_csv(snapshots_path, index=False)
    latest.to_csv(latest_path, index=False)

    date_window = rolling_two_year_quarterly_window()
    metadata = {
        "source": CRICSHEET_ZIP_URL,
        "team_universe": TOP_20_T20I_TEAMS,
        "date_window": {
            "start": str(date_window[0]),
            "end": str(date_window[1]),
        },
        "history_policy": "rolling two-year window over the last eight completed quarters",
        "refresh_cadence": f"every {REFRESH_CADENCE_MONTHS} months",
        "match_rows": int(len(matches)),
        "snapshot_rows": int(len(snapshots)),
        "teams_found": sorted(matches["team"].dropna().unique().tolist()),
    }
    write_json(metadata_path, metadata)
    return DatasetBuildResult(raw_matches_path, snapshots_path, latest_path, metadata_path)
