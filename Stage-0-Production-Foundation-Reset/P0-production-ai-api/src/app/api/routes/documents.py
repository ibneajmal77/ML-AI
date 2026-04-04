from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from app.api.schemas.jobs import JobAcceptedResponse, JobStatusResponse
from app.api.schemas.summarize import ProcessDocumentRequest
from app.config import get_settings
from app.domain.errors import InvalidContentError, NotFoundError
from app.infra.parsers import ContentParser
from app.infra.provider import LocalSummarizerClient
from app.infra.security import rate_limit_request, require_api_key
from app.service.document_service import DocumentService
from app.service.job_service import JobService

router = APIRouter(
    prefix="/v1/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key), Depends(rate_limit_request)],
)


def get_document_service(request: Request) -> DocumentService:
    return DocumentService(
        parser=ContentParser(),
        provider=LocalSummarizerClient(),
        repository=request.app.state.job_repository,
        metrics=request.app.state.metrics,
        settings=get_settings(),
    )


def get_job_service(request: Request) -> JobService:
    return JobService(repository=request.app.state.job_repository)


@router.post("/process", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_document(
    payload: ProcessDocumentRequest,
    background_tasks: BackgroundTasks,
    service: DocumentService = Depends(get_document_service),
) -> JobAcceptedResponse:
    try:
        job_id = service.submit(payload)
    except InvalidContentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    background_tasks.add_task(service.process, job_id, payload)
    return JobAcceptedResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobStatusResponse:
    try:
        job = service.get(job_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        content_type=job.content_type,
        input_size=job.input_size,
        model_name=job.model_name,
        estimated_tokens=job.estimated_tokens,
        estimated_cost_usd=job.estimated_cost_usd,
        summary=job.summary,
        error_message=job.error_message,
    )

