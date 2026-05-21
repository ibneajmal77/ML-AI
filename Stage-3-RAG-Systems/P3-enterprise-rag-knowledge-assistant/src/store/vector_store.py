from __future__ import annotations

import json
import logging
from uuid import UUID

import asyncpg

from src.domain.models import AccessLevel, Chunk, Document, ParentChunk, ScoredChunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_document_by_hash(self, file_hash: str) -> UUID | None:
        row = await self._pool.fetchrow(
            "SELECT id FROM documents WHERE file_hash = $1", file_hash
        )
        return UUID(str(row["id"])) if row else None

    async def upsert_document(self, doc: Document) -> None:
        await self._pool.execute(
            """
            INSERT INTO documents (id, source_url, file_hash, owner_dept, access_level,
                                   created_at, updated_at, tags, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (id) DO UPDATE
                SET updated_at = now(),
                    file_hash  = EXCLUDED.file_hash
            """,
            doc.id,
            doc.source_url,
            doc.file_hash,
            doc.owner_dept,
            int(doc.access_level),
            doc.created_at,
            doc.updated_at,
            doc.tags,
            json.dumps(doc.metadata),
        )

    async def upsert_parent_chunks(self, chunks: list[ParentChunk]) -> list[UUID]:
        saved_ids: list[UUID] = []
        async with self._pool.acquire() as conn:
            for chunk in chunks:
                row = await conn.fetchrow(
                    """
                    INSERT INTO parent_chunks
                        (id, document_id, content, page_number, chunk_index, access_level)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                    """,
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    chunk.page_number,
                    chunk.chunk_index,
                    int(chunk.access_level),
                )
                saved_ids.append(UUID(str(row["id"])) if row else chunk.id)
        return saved_ids

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for chunk in chunks:
                    embedding_str = (
                        f"[{','.join(str(x) for x in chunk.embedding)}]"
                        if chunk.embedding
                        else None
                    )
                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (id, document_id, parent_chunk_id, content, embedding,
                             page_number, chunk_index, token_count, access_level,
                             owner_dept, source_url, tags)
                        VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        chunk.id,
                        chunk.document_id,
                        chunk.parent_chunk_id,
                        chunk.content,
                        embedding_str,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.token_count,
                        int(chunk.access_level),
                        chunk.owner_dept,
                        chunk.source_url,
                        chunk.tags,
                    )

    async def vector_search(
        self,
        query_embedding: list[float],
        user_clearance: int,
        owner_dept: str | None = None,
        top_k: int = 20,
        tags: list[str] | None = None,
    ) -> list[ScoredChunk]:
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        tags_param = tags if tags else None

        if owner_dept:
            rows = await self._pool.fetch(
                """
                SELECT id, document_id, parent_chunk_id, content, page_number,
                       chunk_index, token_count, access_level, owner_dept, source_url, tags,
                       1 - (embedding <=> $1::vector) AS score
                FROM chunks
                WHERE access_level <= $2
                  AND owner_dept = $4
                  AND ($5::text[] IS NULL OR tags && $5::text[])
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                user_clearance,
                top_k,
                owner_dept,
                tags_param,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT id, document_id, parent_chunk_id, content, page_number,
                       chunk_index, token_count, access_level, owner_dept, source_url, tags,
                       1 - (embedding <=> $1::vector) AS score
                FROM chunks
                WHERE access_level <= $2
                  AND ($4::text[] IS NULL OR tags && $4::text[])
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                user_clearance,
                top_k,
                tags_param,
            )

        results = []
        for i, row in enumerate(rows):
            chunk = Chunk(
                id=UUID(str(row["id"])),
                document_id=UUID(str(row["document_id"])),
                parent_chunk_id=UUID(str(row["parent_chunk_id"])) if row["parent_chunk_id"] else None,
                content=row["content"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                access_level=AccessLevel(row["access_level"]),
                owner_dept=row["owner_dept"],
                source_url=row["source_url"],
                tags=list(row["tags"]) if row["tags"] else [],
            )
            results.append(ScoredChunk(chunk=chunk, score=float(row["score"]), rank=i))
        return results

    async def get_parent_content(self, parent_chunk_id: UUID) -> str | None:
        row = await self._pool.fetchrow(
            "SELECT content FROM parent_chunks WHERE id = $1", parent_chunk_id
        )
        return row["content"] if row else None
