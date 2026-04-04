from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.schemas.health import HealthResponse
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", environment=settings.app_env)


@router.get("/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    request.app.state.database.initialize()
    settings = get_settings()
    return HealthResponse(status="ready", environment=settings.app_env)

