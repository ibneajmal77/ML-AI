from __future__ import annotations

from app.domain.models import JobRecord
from app.infra.repositories import JobRepository


class JobService:
    def __init__(self, repository: JobRepository) -> None:
        self._repository = repository

    def get(self, job_id: str) -> JobRecord:
        return self._repository.get_job(job_id)

