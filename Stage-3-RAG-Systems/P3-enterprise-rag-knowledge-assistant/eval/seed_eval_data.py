"""Seed the evaluation database with sample documents matching the golden dataset.

Bypasses the loader (unstructured.io) so this works in CI without system deps.
Reads .txt files directly, chunks them, embeds them, and stores them via VectorStore.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path

import asyncpg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "sample_docs"

DOCS = [
    {
        "file": "support-policy.txt",
        "url": "https://internal/support-policy.pdf",
        "dept": "support",
        "access_level_name": "INTERNAL",
    },
    {
        "file": "data-governance-policy.txt",
        "url": "https://internal/data-governance-policy.pdf",
        "dept": "legal",
        "access_level_name": "INTERNAL",
    },
    {
        "file": "vendor-approval-list.txt",
        "url": "https://internal/vendor-approval-list.pdf",
        "dept": "procurement",
        "access_level_name": "INTERNAL",
    },
    {
        "file": "incident-response-runbook.txt",
        "url": "https://internal/incident-response-runbook.pdf",
        "dept": "engineering",
        "access_level_name": "INTERNAL",
    },
    {
        "file": "sla-agreement.txt",
        "url": "https://internal/sla-agreement.pdf",
        "dept": "support",
        "access_level_name": "INTERNAL",
    },
    {
        "file": "security-access-policy.txt",
        "url": "https://internal/security-access-policy.pdf",
        "dept": "security",
        "access_level_name": "RESTRICTED",
    },
    {
        "file": "hr-access-policy.txt",
        "url": "https://internal/hr-access-policy.pdf",
        "dept": "hr",
        "access_level_name": "CONFIDENTIAL",
    },
]


async def _init_conn(conn: asyncpg.Connection) -> None:
    await conn.execute("SET hnsw.ef_search = 100")


async def seed() -> None:
    from src.config import settings
    from src.domain.models import AccessLevel, Chunk, Document
    from src.embedding.encoder import get_embeddings
    from src.ingestion.chunker import chunk_document
    from src.store.vector_store import VectorStore

    pool = await asyncpg.create_pool(
        settings.database_url, min_size=1, max_size=5, init=_init_conn
    )
    try:
        vector_store = VectorStore(pool)

        for doc_info in DOCS:
            file_path = DOCS_DIR / doc_info["file"]
            text = file_path.read_text(encoding="utf-8")
            access_level = AccessLevel[doc_info["access_level_name"]]

            file_hash = hashlib.sha256(text.encode()).hexdigest()
            existing_id = await vector_store.get_document_by_hash(file_hash)
            if existing_id:
                logger.info("Already seeded (skipping): %s", doc_info["file"])
                continue

            doc = Document(
                source_url=doc_info["url"],
                owner_dept=doc_info["dept"],
                access_level=access_level,
                file_hash=file_hash,
                tags=[],
                metadata={"filename": doc_info["file"]},
            )
            await vector_store.upsert_document(doc)

            _parent_raws, child_raws = chunk_document(
                text,
                strategy="recursive",
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )

            child_texts = [rc.content for rc in child_raws]
            embeddings = await get_embeddings(child_texts, model=settings.embedding_model)

            chunks = [
                Chunk(
                    document_id=doc.id,
                    content=rc.content,
                    embedding=emb,
                    page_number=rc.page_number,
                    chunk_index=rc.chunk_index,
                    token_count=rc.token_count,
                    access_level=access_level,
                    owner_dept=doc_info["dept"],
                    source_url=doc_info["url"],
                    tags=[],
                )
                for rc, emb in zip(child_raws, embeddings)
            ]
            await vector_store.upsert_chunks(chunks)

            logger.info(
                "Seeded %s -> doc_id=%s (%d chunks, access=%s)",
                doc_info["file"],
                doc.id,
                len(chunks),
                doc_info["access_level_name"],
            )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(seed())
