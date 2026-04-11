from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from world_cricket_ml.api.schemas import TeamPrediction
from world_cricket_ml.serving.prediction_service import PredictionService

log = logging.getLogger(__name__)

# Configurable via environment variable so the app works in containers and CI
# without code changes.  Falls back to the repo root derived from this file's
# location when the variable is not set.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(_DEFAULT_ROOT)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the PredictionService once at startup; release on shutdown."""
    log.info("Starting up — PROJECT_ROOT=%s", PROJECT_ROOT)
    app.state.service = PredictionService(PROJECT_ROOT)
    log.info("Startup complete — %d teams loaded", app.state.service.health_status()["teams_loaded"])
    yield
    app.state.service = None
    log.info("Shutdown complete.")


app = FastAPI(title="World Cricket ML Foundations", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return request.app.state.service.health_status()


@app.get("/teams")
def teams(request: Request) -> list[dict]:
    return request.app.state.service.leaderboard()


@app.get("/predict/{team}", response_model=TeamPrediction)
def predict(team: str, request: Request) -> TeamPrediction:
    try:
        return TeamPrediction.model_validate(request.app.state.service.predict_team(team))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown team: {team}") from exc
