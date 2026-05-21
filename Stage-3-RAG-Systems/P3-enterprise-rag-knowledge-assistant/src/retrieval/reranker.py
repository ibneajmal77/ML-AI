from __future__ import annotations

import logging
from functools import lru_cache

import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.domain.models import ScoredChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_cohere_client() -> cohere.AsyncClientV2:
    return cohere.AsyncClientV2(api_key=settings.cohere_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _rerank_with_retry(
    query: str,
    documents: list[str],
    top_n: int,
) -> cohere.RerankResponse:
    client = _get_cohere_client()
    return await client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
    )


async def rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_n: int = 5,
) -> list[ScoredChunk]:
    if not candidates:
        return []

    documents = [c.chunk.content for c in candidates]

    try:
        response = await _rerank_with_retry(query, documents, top_n)
        return [
            ScoredChunk(
                chunk=candidates[r.index].chunk,
                score=r.relevance_score,
                rank=i,
            )
            for i, r in enumerate(response.results)
        ]
    except Exception as exc:
        logger.error("Reranker failed after retries, using RRF ranking: %s", exc)
        return candidates[:top_n]
