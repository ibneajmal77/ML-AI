from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.deps import get_db_pool, get_redis

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready():
    try:
        pool = await get_db_pool()
        await pool.fetchval("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}")

    try:
        redis = await get_redis()
        await redis.ping()
        cache_status = "ok"
    except Exception as exc:
        logger.error("Redis health check failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Cache unreachable: {exc}")

    return {"status": "ready", "db": db_status, "cache": cache_status}
