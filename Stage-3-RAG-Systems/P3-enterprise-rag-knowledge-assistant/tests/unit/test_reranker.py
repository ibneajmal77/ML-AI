from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models import AccessLevel, Chunk, ScoredChunk
from src.retrieval.reranker import rerank


def _make_scored_chunk(content: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content=content,
            page_number=1,
            chunk_index=0,
            token_count=len(content.split()),
            access_level=AccessLevel.INTERNAL,
            owner_dept="test",
            source_url="https://test.com/doc.pdf",
            tags=[],
        ),
        score=score,
        rank=0,
    )


CANDIDATES = [
    _make_scored_chunk("Most relevant content about refund policy", 0.95),
    _make_scored_chunk("Somewhat relevant content about returns", 0.80),
    _make_scored_chunk("Less relevant content about shipping", 0.65),
    _make_scored_chunk("Barely relevant content about FAQ", 0.50),
    _make_scored_chunk("Not very relevant general content", 0.40),
]


@pytest.mark.asyncio
async def test_reranker_fallback_on_error(mocker):
    mocker.patch(
        "src.retrieval.reranker._rerank_with_retry",
        side_effect=Exception("Cohere API error"),
    )

    result = await rerank("test query", CANDIDATES, top_n=3)

    assert len(result) == 3, "Fallback should return top_n candidates"
    assert result[0].chunk.id == CANDIDATES[0].chunk.id, "Fallback preserves original order"


@pytest.mark.asyncio
async def test_reranker_respects_top_n(mocker):
    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.99

    mock_response = MagicMock()
    mock_response.results = [mock_result]

    mocker.patch(
        "src.retrieval.reranker._rerank_with_retry",
        new=AsyncMock(return_value=mock_response),
    )

    result = await rerank("test query", CANDIDATES, top_n=1)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_reranker_preserves_chunk_objects(mocker):
    mock_results = [MagicMock(index=i, relevance_score=1.0 - i * 0.1) for i in range(3)]
    mock_response = MagicMock()
    mock_response.results = mock_results

    mocker.patch(
        "src.retrieval.reranker._rerank_with_retry",
        new=AsyncMock(return_value=mock_response),
    )

    result = await rerank("test query", CANDIDATES, top_n=3)

    for i, sc in enumerate(result):
        original_idx = mock_results[i].index
        assert sc.chunk.id == CANDIDATES[original_idx].chunk.id, (
            "Reranked chunk should be the original Chunk object"
        )
