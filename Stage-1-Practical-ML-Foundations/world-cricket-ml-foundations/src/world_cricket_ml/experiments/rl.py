from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def run_contextual_bandit_proxy(df: pd.DataFrame, artifact_root: Path) -> None:
    frame = df[df["won_toss"] == 1].copy()
    if frame.empty:
        log.warning("Contextual bandit skipped — no toss-winner rows in dataset.")
        write_json(
            artifact_root / "reinforcement_learning" / "contextual_bandit_report.json",
            {"status": "skipped", "reason": "No toss-winner rows available."},
        )
        return

    log.info("Contextual bandit proxy: %d toss-winner rows", len(frame))

    frame["recommended_action"] = frame.apply(
        lambda row: (
            "field"
            if row["opponent_rolling_5_win_rate"] > 0.55
            or row["venue"] in {"Dubai International Cricket Stadium", "Sharjah Cricket Stadium"}
            else "bat"
        ),
        axis=1,
    )
    frame["historical_action"] = frame["toss_decision"].replace({"bowl": "field"})
    frame["policy_match"] = (frame["recommended_action"] == frame["historical_action"]).astype(int)
    frame["reward"] = frame["won_match"]

    historical_rate = round(float(frame.groupby("historical_action")["reward"].mean().mean()), 4)
    policy_rate = round(float(frame.loc[frame["policy_match"] == 1, "reward"].mean()), 4)
    log.info("  historical_reward=%.3f  policy_reward_on_matched=%.3f", historical_rate, policy_rate)

    payload = {
        "status": "completed",
        "historical_reward_rate": historical_rate,
        "recommended_policy_reward_rate_on_matched_contexts": policy_rate,
        "note": (
            "Contextual-bandit-style toss-decisioning exercise.  "
            "This is a heuristic policy evaluation, not a trained RL agent."
        ),
    }
    write_json(artifact_root / "reinforcement_learning" / "contextual_bandit_report.json", payload)
