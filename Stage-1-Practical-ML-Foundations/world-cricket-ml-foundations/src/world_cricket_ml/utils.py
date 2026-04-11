from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_team_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive downfall_risk and surprise_score from model outputs already in *frame*.

    Requires columns: ``dominance_probability``, ``rolling_5_win_rate``,
    ``rolling_10_win_rate``.  Returns a copy with the two signal columns added.
    """
    frame = frame.copy()
    frame["downfall_risk"] = (1 - frame["dominance_probability"]) * frame["rolling_5_win_rate"]
    frame["surprise_score"] = frame["dominance_probability"] * (1 - frame["rolling_10_win_rate"])
    return frame
