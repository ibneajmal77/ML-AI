from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def build_business_framing(latest: pd.DataFrame, artifact_root: Path) -> dict:
    log.info("Building business framing from %d teams", len(latest))
    dominance = latest.sort_values("dominance_probability", ascending=False).head(5)
    downfall = latest.sort_values("downfall_risk", ascending=False).head(5)
    surprise = latest.sort_values("surprise_score", ascending=False).head(5)

    payload = {
        "business_question": "Which top-20 T20I teams are best positioned to dominate the next global event cycle?",
        "decision_outputs": {
            "dominance_watchlist": dominance[["team", "dominance_probability"]].to_dict(orient="records"),
            "downfall_watchlist": downfall[["team", "downfall_risk"]].to_dict(orient="records"),
            "underdog_watchlist": surprise[["team", "surprise_score"]].to_dict(orient="records"),
        },
        "operational_use_cases": [
            "Team analyst dashboard for tournament scouting",
            "Broadcast storyline generation from model-backed momentum signals",
            "Learning project covering end-to-end tabular ML on sports data",
        ],
    }
    log.info(
        "  top_dominant=%s  top_downfall=%s  top_surprise=%s",
        dominance["team"].iloc[0] if not dominance.empty else "n/a",
        downfall["team"].iloc[0] if not downfall.empty else "n/a",
        surprise["team"].iloc[0] if not surprise.empty else "n/a",
    )
    write_json(artifact_root / "business" / "business_decisions.json", payload)
    return payload
