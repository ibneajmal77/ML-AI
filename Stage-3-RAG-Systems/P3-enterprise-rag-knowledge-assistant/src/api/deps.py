from __future__ import annotations

import logging

import asyncpg
from redis.asyncio import Redis, from_url

from src.config import settings
from src.cache.query_cache import QueryCache
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.store.bm25_index import BM25Index
from src.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_bm25: BM25Index | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.execute("SET hnsw.ef_search = 100")


async def get_db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            init=_init_connection,
        )
    return _pool


async def get_vector_store() -> VectorStore:
    pool = await get_db_pool()
    return VectorStore(pool)


async def get_bm25_index() -> BM25Index:
    global _bm25
    if _bm25 is None:
        pool = await get_db_pool()
        _bm25 = BM25Index()
        await _bm25.build_from_db(pool)
        logger.info("BM25 index ready")
    return _bm25


async def get_hybrid_retriever() -> HybridRetriever:
    vs = await get_vector_store()
    bm25 = await get_bm25_index()
    return HybridRetriever(vs, bm25)


async def get_redis() -> Redis:
    return from_url(settings.redis_url, decode_responses=True)


async def get_query_cache() -> QueryCache:
    redis = await get_redis()
    return QueryCache(redis)


async def get_ingestion_pipeline() -> IngestionPipeline:
    pool = await get_db_pool()
    vs = VectorStore(pool)
    bm25 = await get_bm25_index()
    return IngestionPipeline(pool, vs, bm25)
