from __future__ import annotations

import logging
from uuid import UUID

import asyncpg

from src.config import settings
from src.domain.models import AccessLevel, Chunk, Document, DocumentMeta, ParentChunk
from src.embedding.encoder import get_embeddings
from src.ingestion.chunker import RawChunk, chunk_document
from src.ingestion.loader import load_document
from src.store.bm25_index import BM25Index
from src.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        pool: asyncpg.Pool,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ) -> None:
        self._pool = pool
        self._vector_store = vector_store
        self._bm25 = bm25_index

    async def run(
        self,
        file_bytes: bytes,
        filename: str,
        meta: DocumentMeta,
        chunking_strategy: str = "recursive",
        use_azure_di: bool = False,
    ) -> UUID:
        # 1. Load document
        raw_doc = await load_document(file_bytes, filename, use_azure_di=use_azure_di)

        # 2. Dedup by file hash
        existing_id = await self._vector_store.get_document_by_hash(raw_doc.file_hash)
        if existing_id:
            logger.info("Document already indexed (hash match), skipping: %s", existing_id)
            return existing_id

        # 3. Create and store document record
        doc = Document(
            source_url=meta.source_url,
            owner_dept=meta.owner_dept,
            access_level=meta.access_level,
            tags=meta.tags,
            metadata={**meta.metadata, **raw_doc.metadata},
            file_hash=raw_doc.file_hash,
        )
        try:
            await self._vector_store.upsert_document(doc)
        except asyncpg.exceptions.UniqueViolationError:
            # Race condition: another worker inserted the same hash concurrently
            existing_id = await self._vector_store.get_document_by_hash(raw_doc.file_hash)
            if existing_id:
                logger.info("Document already indexed (concurrent insert), skipping: %s", existing_id)
                return existing_id
            raise
        logger.info("Document created: %s (%s)", doc.id, filename)

        # 4. Chunk the document
        parent_raws, child_raws = chunk_document(
            raw_doc.text,
            strategy=chunking_strategy,
            chunk_size=settings.chunk_size,
            parent_chunk_size=settings.parent_chunk_size,
            overlap=settings.chunk_overlap,
        )

        # 5. Store parent chunks
        parent_chunks = [
            ParentChunk(
                document_id=doc.id,
                content=rc.content,
                page_number=rc.page_number,
                chunk_index=rc.chunk_index,
                access_level=meta.access_level,
            )
            for rc in parent_raws
        ]
        parent_ids = await self._vector_store.upsert_parent_chunks(parent_chunks)

        # 6. Embed child chunks
        child_texts = [rc.content for rc in child_raws]
        embeddings = await get_embeddings(child_texts, model=settings.embedding_model)

        # 7. Build and store child chunks
        chunks: list[Chunk] = []
        for rc, embedding in zip(child_raws, embeddings):
            parent_chunk_id: UUID | None = None
            if rc.parent_index is not None and rc.parent_index < len(parent_ids):
                parent_chunk_id = parent_ids[rc.parent_index]

            chunk = Chunk(
                document_id=doc.id,
                parent_chunk_id=parent_chunk_id,
                content=rc.content,
                embedding=embedding,
                page_number=rc.page_number,
                chunk_index=rc.chunk_index,
                token_count=rc.token_count,
                access_level=meta.access_level,
                owner_dept=meta.owner_dept,
                source_url=meta.source_url,
                tags=meta.tags,
            )
            chunks.append(chunk)

        await self._vector_store.upsert_chunks(chunks)
        logger.info("Stored %d chunks for document %s", len(chunks), doc.id)

        # 8. Update BM25 index
        chunk_ids = [str(c.id) for c in chunks]
        await self._bm25.add_documents(chunk_ids, child_texts)

        return doc.id
