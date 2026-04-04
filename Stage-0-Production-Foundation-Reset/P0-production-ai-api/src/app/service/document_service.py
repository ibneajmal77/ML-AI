from __future__ import annotations

import time
from uuid import uuid4

from app.api.schemas.summarize import ProcessDocumentRequest
from app.config import Settings
from app.domain.errors import InvalidContentError
from app.infra.metrics import InMemoryMetrics
from app.infra.parsers import ContentParser
from app.infra.provider import LocalSummarizerClient
from app.infra.repositories import JobRepository
from app.service.costing import estimate_cost_usd, estimate_tokens


class DocumentService:
    def __init__(
        self,
        *,
        parser: ContentParser,
        provider: LocalSummarizerClient,
        repository: JobRepository,
        metrics: InMemoryMetrics,
        settings: Settings,
    ) -> None:
        self._parser = parser
        self._provider = provider
        self._repository = repository
        self._metrics = metrics
        self._settings = settings

    def submit(self, request: ProcessDocumentRequest) -> str:
        if request.content_type not in self._parser.supported_content_types:
            raise InvalidContentError(f"Unsupported content type: {request.content_type}")

        estimated_tokens = estimate_tokens(request.content)
        estimated_cost_usd = estimate_cost_usd(estimated_tokens)
        job = self._repository.create_job(
            job_id=str(uuid4()),
            status="queued",
            content_type=request.content_type,
            input_size=len(request.content),
            model_name=self._settings.default_model_name,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        self._metrics.increment("document_jobs_submitted_total")
        return job.job_id

    def process(self, job_id: str, request: ProcessDocumentRequest) -> None:
        self._repository.update_job(job_id, status="running")
        try:
            parsed_content = self._parser.parse(request.content, request.content_type)
            time.sleep(self._settings.job_work_simulation_seconds)
            summary = self._provider.summarize(parsed_content, request.max_sentences)
            self._repository.update_job(job_id, status="completed", summary=summary)
            self._metrics.increment("document_jobs_completed_total")
        except Exception as exc:
            self._repository.update_job(job_id, status="failed", error_message=str(exc))
            self._metrics.increment("document_jobs_failed_total")

