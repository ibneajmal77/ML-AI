from __future__ import annotations

import pytest

from src.domain.models import AccessLevel, DocumentMeta
from src.ingestion.pipeline import IngestionPipeline
from src.store.bm25_index import BM25Index
from src.store.vector_store import VectorStore

SAMPLE_TEXT_BYTES = b"""This is a sample document for testing the ingestion pipeline.
It contains multiple sentences across several paragraphs.

The second paragraph discusses enterprise knowledge management.
Documents are chunked, embedded, and stored for retrieval.

The third paragraph covers access control policies.
Users must have appropriate clearance to access restricted content."""

SAMPLE_PDF_BYTES = SAMPLE_TEXT_BYTES


@pytest.fixture
async def pipeline(pg_pool):
    vs = VectorStore(pg_pool)
    bm25 = BM25Index()
    return IngestionPipeline(pg_pool, vs, bm25)


@pytest.mark.asyncio
async def test_full_ingest_stores_chunks(pipeline, pg_pool):
    meta = DocumentMeta(
        source_url="https://test.com/sample.txt",
        owner_dept="engineering",
        access_level=AccessLevel.INTERNAL,
        tags=["test"],
    )

    doc_id = await pipeline.run(
        SAMPLE_TEXT_BYTES, "sample.txt", meta, chunking_strategy="recursive"
    )

    count = await pg_pool.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE document_id = $1", doc_id
    )
    assert count > 0, f"Expected chunks for document {doc_id}"


@pytest.mark.asyncio
async def test_dedup_on_same_file_hash(pipeline, pg_pool):
    meta = DocumentMeta(
        source_url="https://test.com/dedup-test.txt",
        owner_dept="engineering",
        access_level=AccessLevel.INTERNAL,
    )

    doc_id_1 = await pipeline.run(SAMPLE_TEXT_BYTES, "dedup.txt", meta)
    doc_id_2 = await pipeline.run(SAMPLE_TEXT_BYTES, "dedup.txt", meta)

    assert doc_id_1 == doc_id_2, "Same file hash should return existing document ID"

    doc_count = await pg_pool.fetchval(
        "SELECT COUNT(*) FROM documents WHERE id = $1", doc_id_1
    )
    assert doc_count == 1, "Only one document row should exist for deduplicated file"


@pytest.mark.asyncio
async def test_access_level_stored_correctly(pipeline, pg_pool):
    meta = DocumentMeta(
        source_url="https://test.com/restricted.txt",
        owner_dept="legal",
        access_level=AccessLevel.RESTRICTED,
        tags=["confidential"],
    )

    unique_bytes = SAMPLE_TEXT_BYTES + b"\nUnique content for restricted test."
    doc_id = await pipeline.run(unique_bytes, "restricted.txt", meta)

    access_level = await pg_pool.fetchval(
        "SELECT access_level FROM documents WHERE id = $1", doc_id
    )
    assert access_level == int(AccessLevel.RESTRICTED), (
        f"Expected access_level={int(AccessLevel.RESTRICTED)}, got {access_level}"
    )
