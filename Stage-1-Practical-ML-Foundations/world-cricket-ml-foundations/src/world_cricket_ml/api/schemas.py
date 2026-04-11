from __future__ import annotations

from pydantic import BaseModel


class TeamPrediction(BaseModel):
    team: str
    dominance_probability: float
    predicted_future_win_rate: float
    downfall_risk: float
    surprise_score: float
    recent_form: dict[str, float]
