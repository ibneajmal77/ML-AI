from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: str
    content_type: str
    input_size: int
    model_name: str
    estimated_tokens: int
    estimated_cost_usd: float
    summary: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

