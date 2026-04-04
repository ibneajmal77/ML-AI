from __future__ import annotations

from pydantic import BaseModel


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    content_type: str | None = None
    input_size: int | None = None
    model_name: str | None = None
    estimated_tokens: int | None = None
    estimated_cost_usd: float | None = None
    summary: str | None = None
    error_message: str | None = None

