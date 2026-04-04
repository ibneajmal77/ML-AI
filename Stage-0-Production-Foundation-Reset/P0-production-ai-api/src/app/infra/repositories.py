from __future__ import annotations

from datetime import datetime, timezone

from app.domain.errors import NotFoundError
from app.domain.models import JobRecord
from app.infra.db import Database


class JobRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create_job(
        self,
        *,
        job_id: str,
        status: str,
        content_type: str,
        input_size: int,
        model_name: str,
        estimated_tokens: int,
        estimated_cost_usd: float,
    ) -> JobRecord:
        now = datetime.now(timezone.utc)
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, content_type, input_size, model_name,
                    estimated_tokens, estimated_cost_usd, summary, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    status,
                    content_type,
                    input_size,
                    model_name,
                    estimated_tokens,
                    estimated_cost_usd,
                    None,
                    None,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            connection.commit()
        return self.get_job(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        summary: str | None = None,
        error_message: str | None = None,
    ) -> JobRecord:
        now = datetime.now(timezone.utc)
        with self._database.connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, summary = ?, error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, summary, error_message, now.isoformat(), job_id),
            )
            connection.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Job {job_id} not found.")
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            content_type=row["content_type"],
            input_size=row["input_size"],
            model_name=row["model_name"],
            estimated_tokens=row["estimated_tokens"],
            estimated_cost_usd=row["estimated_cost_usd"],
            summary=row["summary"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

