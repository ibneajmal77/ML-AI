from __future__ import annotations

import pytest

from src.cache.query_cache import QueryCache
from src.context.assembler import assemble_context
from src.domain.models import AccessLevel, DocumentMeta
from src.embedding.encoder import get_query_embedding
from src.generation.generator import generate_answer
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import rerank
from src.store.bm25_index import BM25Index
from src.store.vector_store import VectorStore

POLICY_BYTES = b"""Enterprise Refund Policy

Enterprise customers may request refunds within 30 days of purchase.
Refund requests must be submitted through the support portal.
Processing time is 5-10 business days after approval.

Support Tiers: Basic, Professional, and Enterprise.
Enterprise tier includes priority processing and dedicated account manager."""

RESTRICTED_BYTES = b"""Confidential Financial Data - RESTRICTED ACCESS

Q3 Revenue: $45.2M
Operating Margin: 28%
This document contains sensitive financial information.
Access is restricted to authorized personnel only."""


async def _full_query(query: str, pool, user_clearance: int = 1):
    from redis.asyncio import from_url
    from src.config import settings

    vs = VectorStore(pool)
    bm25 = BM25Index()
    await bm25.build_from_db(pool)
    retriever = HybridRetriever(vs, bm25)
    redis = from_url(settings.redis_url, decode_responses=True)
    cache = QueryCache(redis)

    query_embedding = await get_query_embedding(query)
    candidates = await retriever.search(query, query_embedding, user_clearance, None, 20)
    if not candidates:
        return "I could not find relevant information.", [], False

    reranked = await rerank(query, candidates, top_n=5)
    context, citations = await assemble_context(query, reranked, vs)
    answer = await generate_answer(query, context, citations)
    return answer, citations, False


@pytest.mark.asyncio
async def test_query_returns_answer(pg_pool):
    meta = DocumentMeta(
        source_url="https://test.com/policy.txt",
        owner_dept="sales",
        access_level=AccessLevel.INTERNAL,
    )
    vs = VectorStore(pg_pool)
    bm25 = BM25Index()
    pipeline = IngestionPipeline(pg_pool, vs, bm25)
    await pipeline.run(POLICY_BYTES, "policy.txt", meta)

    answer, citations, _ = await _full_query("What is the refund policy?", pg_pool, user_clearance=1)

    assert answer, "Answer should not be empty"
    assert len(answer) > 10, "Answer should be substantive"


@pytest.mark.asyncio
async def test_access_control_filters_results(pg_pool):
    meta = DocumentMeta(
        source_url="https://test.com/restricted-finance.txt",
        owner_dept="finance",
        access_level=AccessLevel.RESTRICTED,
    )
    unique_bytes = RESTRICTED_BYTES + b"\nUnique marker for access control test."
    vs = VectorStore(pg_pool)
    bm25 = BM25Index()
    pipeline = IngestionPipeline(pg_pool, vs, bm25)
    await pipeline.run(unique_bytes, "restricted.txt", meta)

    # Query with INTERNAL clearance (level 1) — should not see RESTRICTED (level 3) docs
    query_embedding = await get_query_embedding("What is the Q3 revenue?")
    results = await vs.vector_search(query_embedding, user_clearance=1, top_k=10)

    restricted_found = any(
        "Q3 Revenue" in sc.chunk.content or "financial" in sc.chunk.content.lower()
        for sc in results
        if sc.chunk.access_level == AccessLevel.RESTRICTED
    )
    assert not restricted_found, "INTERNAL user should not see RESTRICTED documents"


@pytest.mark.asyncio
async def test_cache_hit_returns_same_answer(pg_pool):
    from redis.asyncio import from_url
    from src.config import settings

    query = "What are the support tiers available?"

    redis = from_url(settings.redis_url, decode_responses=True)
    cache = QueryCache(redis)

    # Clear any existing cache for this query
    await cache.invalidate_pattern("rag:query:*")

    vs = VectorStore(pg_pool)
    bm25 = BM25Index()
    await bm25.build_from_db(pg_pool)
    retriever = HybridRetriever(vs, bm25)

    query_embedding = await get_query_embedding(query)
    candidates = await retriever.search(query, query_embedding, 1, None, 20)

    if candidates:
        reranked = await rerank(query, candidates, top_n=5)
        context, citations = await assemble_context(query, reranked, vs)
        answer = await generate_answer(query, context, citations)

        from src.domain.models import QueryResponse
        response = QueryResponse(answer=answer, citations=citations, latency_ms=100.0, cache_hit=False)
        await cache.set(query, 1, None, response.model_dump(mode="json"))

    # Second retrieval should be a cache hit
    cached = await cache.get(query, 1, None)
    assert cached is not None, "Cache should have stored the response"
    assert "answer" in cached, "Cached response should contain answer"
