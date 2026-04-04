from __future__ import annotations

from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    payload = request.app.state.metrics.render_prometheus()
    return Response(content=payload, media_type="text/plain; version=0.0.4")

