from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ACTIONS = ["do_nothing", "send_email", "offer_discount"]


def _state_bucket(row: pd.Series) -> str:
    if row["days_since_last_activity"] > 25 and row["tickets_30d"] >= 4:
        return "high_risk"
    if row["feature_adoption_ratio"] > 0.7 and row["nps_score"] > 20:
        return "healthy"
    return "medium_risk"


def _reward(state: str, action: str, rng: np.random.Generator) -> float:
    base = {
        ("high_risk", "do_nothing"): -3.0,
        ("high_risk", "send_email"): 0.8,
        ("high_risk", "offer_discount"): 1.5,
        ("medium_risk", "do_nothing"): 0.3,
        ("medium_risk", "send_email"): 0.7,
        ("medium_risk", "offer_discount"): 0.2,
        ("healthy", "do_nothing"): 0.9,
        ("healthy", "send_email"): 0.2,
        ("healthy", "offer_discount"): -0.8,
    }[(state, action)]
    return float(base + rng.normal(0, 0.2))


def run_contextual_bandit(df: pd.DataFrame, output_dir: Path, episodes: int = 2000, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    q_values = {state: {action: 0.0 for action in ACTIONS} for state in ["high_risk", "medium_risk", "healthy"]}
    counts = {state: {action: 0 for action in ACTIONS} for state in ["high_risk", "medium_risk", "healthy"]}
    rewards: list[float] = []

    sampled = df.sample(n=min(len(df), episodes), replace=True, random_state=seed).reset_index(drop=True)
    epsilon = 0.15
    for _, row in sampled.iterrows():
        state = _state_bucket(row)
        if rng.random() < epsilon:
            action = rng.choice(ACTIONS)
        else:
            action = max(q_values[state], key=q_values[state].get)
        reward = _reward(state, action, rng)
        counts[state][action] += 1
        step = counts[state][action]
        q_values[state][action] += (reward - q_values[state][action]) / step
        rewards.append(reward)

    payload = {
        "policy": {state: max(actions, key=actions.get) for state, actions in q_values.items()},
        "q_values": q_values,
        "average_reward": round(float(np.mean(rewards)), 4),
        "episodes": len(rewards),
    }
    (output_dir / "contextual_bandit_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
