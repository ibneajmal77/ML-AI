from __future__ import annotations

import logging
from collections import defaultdict

from src.domain.models import ScoredChunk
from src.store.bm25_index import BM25Index
from src.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[ScoredChunk],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[ScoredChunk]:
    """Merge dense and BM25 results using Reciprocal Rank Fusion.

    RRF formula: score(doc) = Σ 1 / (k + rank_i), summed across all lists.
    Only chunks present in dense_results are returned (BM25-only chunks lack
    full Chunk objects and are excluded — design decision for precision).
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, ScoredChunk] = {}

    for rank, sc in enumerate(dense_results):
        chunk_id = str(sc.chunk.id)
        rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
        chunk_map[chunk_id] = sc

    for rank, (chunk_id, _score) in enumerate(bm25_results):
        rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    result: list[ScoredChunk] = []
    for final_rank, chunk_id in enumerate(sorted_ids):
        if chunk_id in chunk_map:
            sc = chunk_map[chunk_id]
            result.append(
                ScoredChunk(chunk=sc.chunk, score=rrf_scores[chunk_id], rank=final_rank)
            )

    return result


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25_index

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        user_clearance: int,
        owner_dept: str | None = None,
        top_k: int = 20,
        tags: list[str] | None = None,
    ) -> list[ScoredChunk]:
        from src.config import settings

        dense_results = await self._vector_store.vector_search(
            query_embedding, user_clearance, owner_dept, top_k, tags
        )
        bm25_results = self._bm25.search(query, top_k)

        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=settings.rrf_k)
        return fused[:top_k]
