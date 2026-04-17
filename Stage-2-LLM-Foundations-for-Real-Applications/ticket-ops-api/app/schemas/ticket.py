from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class ClassifyResponse(BaseModel):
    label: Literal["billing", "technical", "account", "general"]
    input_tokens: int
    output_tokens: int
    prompt_version: str


class TicketExtraction(BaseModel):
    issue_type: str | None = None
    urgency: Literal["low", "medium", "high"] | None = None
    account_id: str | None = None
    submitted_at: str | None = None

    @field_validator("submitted_at")
    @classmethod
    def validate_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return value
        date.fromisoformat(value)
        return value


class ExtractResponse(BaseModel):
    data: TicketExtraction
    input_tokens: int
    output_tokens: int
    prompt_version: str
