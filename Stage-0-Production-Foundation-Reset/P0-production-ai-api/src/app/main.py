from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.summarize import router as summarize_router
from app.config import get_settings
from app.infra.db import Database
from app.infra.logging import configure_logging
from app.infra.metrics import InMemoryMetrics
from app.infra.rate_limit import InMemoryRateLimiter
from app.infra.repositories import JobRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()
    database = Database(settings.database_path)
    database.initialize()
    app.state.database = database
    app.state.job_repository = JobRepository(database)
    app.state.metrics = InMemoryMetrics()
    app.state.rate_limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="P0 Production AI API", debug=get_settings().debug, lifespan=lifespan)

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(summarize_router)
    app.include_router(documents_router)

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        correlation_id = request.headers.get("x-correlation-id", str(uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["x-correlation-id"] = correlation_id
        return response

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        logger = logging.getLogger("app.request")
        response = await call_next(request)
        logger.info(
            "request_completed",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", "missing"),
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                },
            },
        )
        return response

    return app


app = create_app()
