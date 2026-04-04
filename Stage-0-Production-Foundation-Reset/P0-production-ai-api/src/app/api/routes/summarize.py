from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.schemas.summarize import SummarizeRequest, SummarizeResponse
from app.config import get_settings
from app.infra.provider import LocalSummarizerClient
from app.infra.security import rate_limit_request, require_api_key
from app.service.summarize_service import SummarizeService

router = APIRouter(prefix="/v1", tags=["summarize"], dependencies=[Depends(require_api_key), Depends(rate_limit_request)])


def get_summarize_service(request: Request) -> SummarizeService:
    return SummarizeService(
        provider=LocalSummarizerClient(),
        metrics=request.app.state.metrics,
        settings=get_settings(),
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    payload: SummarizeRequest,
    service: SummarizeService = Depends(get_summarize_service),
) -> SummarizeResponse:
    return SummarizeResponse.model_validate(await service.summarize(payload))

