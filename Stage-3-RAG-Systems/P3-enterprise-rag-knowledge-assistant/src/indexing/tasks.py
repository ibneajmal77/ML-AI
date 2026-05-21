from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from src.domain.models import AccessLevel, DocumentMeta
from src.indexing.worker import celery_app

logger = logging.getLogger(__name__)


async def _run_ingestion(
    file_bytes: bytes,
    filename: str,
    meta: DocumentMeta,
    chunking_strategy: str,
    use_azure_di: bool,
) -> str:
    import asyncpg
    from src.config import settings
    from src.ingestion.pipeline import IngestionPipeline
    from src.store.bm25_index import BM25Index
    from src.store.vector_store import VectorStore

    async def _init_conn(conn):
        await conn.execute("SET hnsw.ef_search = 100")

    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=3, init=_init_conn)
    try:
        vs = VectorStore(pool)
        bm25 = BM25Index()
        await bm25.build_from_db(pool)
        pipeline = IngestionPipeline(pool, vs, bm25)
        doc_id = await pipeline.run(file_bytes, filename, meta, chunking_strategy, use_azure_di)
        return str(doc_id)
    finally:
        await pool.close()


@celery_app.task(bind=True, max_retries=3)
def ingest_document_task(
    self,
    file_path: str,
    source_url: str,
    owner_dept: str,
    access_level: int = 1,
    tags: list[str] | None = None,
    chunking_strategy: str = "recursive",
    use_azure_di: bool = False,
) -> str:
    try:
        file_bytes = Path(file_path).read_bytes()
        filename = Path(file_path).name
        meta = DocumentMeta(
            source_url=source_url,
            owner_dept=owner_dept,
            access_level=AccessLevel(access_level),
            tags=tags or [],
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            doc_id = loop.run_until_complete(
                _run_ingestion(file_bytes, filename, meta, chunking_strategy, use_azure_di)
            )
        finally:
            loop.close()
        logger.info("Ingested document: %s -> %s", file_path, doc_id)
        return doc_id
    except Exception as exc:
        logger.error("Ingestion task failed: %s", exc)
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def incremental_sync_task(self, watch_dir: str = "/app/docs_watch") -> dict:
    async def _run():
        import asyncpg
        from src.config import settings

        pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=2)
        try:
            rows = await pool.fetch("SELECT source_url, file_hash FROM documents")
            known: dict[str, str] = {row["source_url"]: row["file_hash"] for row in rows}
        finally:
            await pool.close()
        return known

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            known = loop.run_until_complete(_run())
        finally:
            loop.close()

        queued = 0
        watch_path = Path(watch_dir)
        if not watch_path.exists():
            return {"queued": 0, "message": f"watch_dir {watch_dir} does not exist"}

        for file_path in watch_path.rglob("*"):
            if file_path.is_dir():
                continue
            file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            source_url = str(file_path)
            if source_url not in known or known[source_url] != file_hash:
                ingest_document_task.delay(
                    file_path=str(file_path),
                    source_url=source_url,
                    owner_dept="sync",
                )
                queued += 1

        logger.info("Incremental sync: queued %d files", queued)
        return {"queued": queued}
    except Exception as exc:
        logger.error("Incremental sync failed: %s", exc)
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def full_reindex_task(self) -> dict:
    async def _run():
        import asyncpg
        from src.config import settings
        from src.embedding.encoder import get_embeddings

        async def _init_conn(conn):
            await conn.execute("SET hnsw.ef_search = 100")

        pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=3, init=_init_conn)
        try:
            rows = await pool.fetch("SELECT id::text, content FROM chunks ORDER BY created_at")
            if not rows:
                return {"reindexed": 0}

            ids = [row["id"] for row in rows]
            texts = [row["content"] for row in rows]

            embeddings = await get_embeddings(texts, model=settings.embedding_model)

            async with pool.acquire() as conn:
                async with conn.transaction():
                    for chunk_id, embedding in zip(ids, embeddings):
                        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                        await conn.execute(
                            "UPDATE chunks SET embedding = $1::vector WHERE id = $2::uuid",
                            embedding_str,
                            chunk_id,
                        )

            return {"reindexed": len(ids)}
        finally:
            await pool.close()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()
        logger.info("Full reindex complete: %s", result)
        return result
    except Exception as exc:
        logger.error("Full reindex failed: %s", exc)
        raise self.retry(exc=exc, countdown=300)
