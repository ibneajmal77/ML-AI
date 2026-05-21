from __future__ import annotations

import uuid

import pytest

from src.domain.models import AccessLevel, Chunk, ScoredChunk
from src.retrieval.hybrid import reciprocal_rank_fusion


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=uuid.UUID(chunk_id),
        document_id=uuid.uuid4(),
        content=f"Content for {chunk_id}",
        page_number=1,
        chunk_index=0,
        token_count=10,
        access_level=AccessLevel.INTERNAL,
        owner_dept="test",
        source_url="https://test.com/doc.pdf",
        tags=[],
    )


ID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
ID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
ID_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def test_rrf_boosts_document_in_both_lists():
    chunk_a = _make_chunk(ID_A)
    chunk_b = _make_chunk(ID_B)

    dense = [ScoredChunk(chunk=chunk_a, score=0.9, rank=0), ScoredChunk(chunk=chunk_b, score=0.8, rank=1)]
    bm25 = [(ID_A, 5.0), (ID_B, 2.0)]

    # Boost both by appearing in dense too
    dense_only = [ScoredChunk(chunk=chunk_b, score=0.8, rank=0)]
    bm25_only = [(ID_B, 2.0)]

    result_both = reciprocal_rank_fusion(dense, bm25)
    result_single = reciprocal_rank_fusion(dense_only, bm25_only)

    # A appeared in both lists, B only in one pair — A should score higher
    score_a = next(sc.score for sc in result_both if str(sc.chunk.id) == ID_A)
    score_b_both = next(sc.score for sc in result_both if str(sc.chunk.id) == ID_B)
    score_b_single = result_single[0].score

    assert score_a > score_b_single, "A in both lists should outscore B in only one list"
    assert score_b_both > score_b_single, "B in both lists should outscore B in one list"


def test_rrf_k60_dampening():
    chunk_a = _make_chunk(ID_A)
    chunk_b = _make_chunk(ID_B)

    dense = [ScoredChunk(chunk=chunk_a, score=0.9, rank=0), ScoredChunk(chunk=chunk_b, score=0.5, rank=9)]
    bm25 = []

    result = reciprocal_rank_fusion(dense, bm25, k=60)
    score_rank0 = next(sc.score for sc in result if str(sc.chunk.id) == ID_A)
    score_rank9 = next(sc.score for sc in result if str(sc.chunk.id) == ID_B)

    expected_rank0 = 1.0 / (60 + 0 + 1)
    expected_rank9 = 1.0 / (60 + 9 + 1)
    assert abs(score_rank0 - expected_rank0) < 1e-9
    assert abs(score_rank9 - expected_rank9) < 1e-9


def test_rrf_returns_only_chunks_from_dense_results():
    chunk_a = _make_chunk(ID_A)
    dense = [ScoredChunk(chunk=chunk_a, score=0.9, rank=0)]
    bm25 = [(ID_A, 3.0), (ID_B, 5.0)]  # ID_B not in dense

    result = reciprocal_rank_fusion(dense, bm25)
    result_ids = {str(sc.chunk.id) for sc in result}

    assert ID_A in result_ids, "A should be in results (in dense)"
    assert ID_B not in result_ids, "B should be excluded (BM25-only, no full Chunk object)"


def test_rrf_empty_bm25():
    chunk_a = _make_chunk(ID_A)
    dense = [ScoredChunk(chunk=chunk_a, score=0.9, rank=0)]
    bm25: list = []

    result = reciprocal_rank_fusion(dense, bm25)
    assert len(result) == 1
    assert str(result[0].chunk.id) == ID_A


def test_rrf_empty_dense():
    bm25 = [(ID_A, 5.0)]
    result = reciprocal_rank_fusion([], bm25)
    assert result == [], "Empty dense results should produce empty output"
