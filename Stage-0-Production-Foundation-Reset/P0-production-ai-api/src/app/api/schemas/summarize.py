from __future__ import annotations

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    max_sentences: int = Field(default=2, ge=1, le=5)


class SummarizeResponse(BaseModel):
    summary: str
    model_name: str
    estimated_tokens: int
    estimated_cost_usd: float
    processing_mode: str


class ProcessDocumentRequest(BaseModel):
    content: str = Field(min_length=1, max_length=200_000)
    content_type: str = Field(default="text/plain")
    max_sentences: int = Field(default=3, ge=1, le=6)

