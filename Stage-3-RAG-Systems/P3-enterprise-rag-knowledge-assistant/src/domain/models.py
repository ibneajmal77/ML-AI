from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AccessLevel(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


class DocumentMeta(BaseModel):
    source_url: str
    owner_dept: str
    access_level: AccessLevel = AccessLevel.INTERNAL
    tags: list[str] = []
    metadata: dict = {}


class Document(DocumentMeta):
    id: UUID = Field(default_factory=uuid4)
    file_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ParentChunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    page_number: int | None
    chunk_index: int
    access_level: AccessLevel


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    parent_chunk_id: UUID | None = None
    content: str
    embedding: list[float] | None = None
    page_number: int | None = None
    chunk_index: int
    token_count: int
    access_level: AccessLevel
    owner_dept: str
    source_url: str
    tags: list[str] = []


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int = 0


class Citation(BaseModel):
    chunk_id: UUID
    source_url: str
    page_number: int | None
    excerpt: str


class IngestRequest(BaseModel):
    source_url: str
    owner_dept: str
    access_level: AccessLevel = AccessLevel.INTERNAL
    tags: list[str] = []
    chunking_strategy: str = "recursive"
    use_azure_di: bool = False


class IngestResponse(BaseModel):
    document_id: UUID
    task_id: str
    status: str = "queued"


class QueryRequest(BaseModel):
    query: str
    user_clearance: AccessLevel = AccessLevel.INTERNAL
    owner_dept: str | None = None
    tags: list[str] = []
    use_hyde: bool = False
    use_multi_query: bool = False
    conversation_history: list[dict] = []


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    faithfulness_hint: str | None = None
    latency_ms: float
    cache_hit: bool = False
