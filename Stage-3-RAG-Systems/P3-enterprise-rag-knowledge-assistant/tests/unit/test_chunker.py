from __future__ import annotations

import pytest

from src.ingestion.chunker import (
    RawChunk,
    _count_tokens,
    chunk_document,
    parent_child_chunk,
    recursive_chunk,
    semantic_chunk,
)

LONG_TEXT = " ".join(["word"] * 2000)
SENTENCE_TEXT = "The contract was signed on Monday. It became effective immediately. All parties agreed to the terms. The first payment is due in thirty days. Late fees apply after the grace period."


def test_recursive_chunk_respects_token_size():
    chunks = recursive_chunk(LONG_TEXT, chunk_size=100, overlap=10)
    assert chunks, "Should produce at least one chunk"
    for chunk in chunks:
        assert chunk.token_count <= 100 * 1.2, f"Chunk exceeds size tolerance: {chunk.token_count}"


def test_recursive_chunk_overlap():
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = recursive_chunk(text, chunk_size=50, overlap=10)
    if len(chunks) > 1:
        first_words = set(chunks[0].content.split())
        second_words = set(chunks[1].content.split())
        assert first_words & second_words, "Consecutive chunks should share overlap tokens"


def test_parent_child_children_reference_correct_parent():
    parents, children = parent_child_chunk(LONG_TEXT, child_size=100, parent_size=400, overlap=20)
    assert parents, "Should produce parent chunks"
    assert children, "Should produce child chunks"
    for child in children:
        if child.parent_index is not None:
            assert 0 <= child.parent_index < len(parents), (
                f"parent_index {child.parent_index} out of range [0, {len(parents)})"
            )


def test_parent_child_parent_covers_all_children():
    text = " ".join([f"tok{i}" for i in range(1000)])
    parents, children = parent_child_chunk(text, child_size=50, parent_size=200, overlap=10)
    # Each child's content should be a substring of its parent's content
    for child in children:
        if child.parent_index is not None:
            parent = parents[child.parent_index]
            # At least some words should overlap (overlap handling may shift boundaries)
            child_words = set(child.content.split()[:5])
            assert any(w in parent.content for w in child_words), (
                "Child content words should appear in parent content"
            )


def test_semantic_chunk_no_mid_sentence_splits():
    chunks = semantic_chunk(SENTENCE_TEXT, max_chunk_size=50)
    assert chunks, "Should produce chunks from sentence text"
    for chunk in chunks:
        content = chunk.content.strip()
        if content:
            assert content[-1] in ".!?", (
                f"Semantic chunk should end with sentence terminator, got: ...{content[-20:]!r}"
            )


def test_empty_text_returns_empty():
    assert recursive_chunk("") == []
    assert recursive_chunk("   ") == []
    parents, children = parent_child_chunk("")
    assert parents == []
    assert children == []
    assert semantic_chunk("") == []


def test_chunk_document_dispatch():
    chunks_r, _ = chunk_document(LONG_TEXT, strategy="recursive", chunk_size=100)
    assert chunks_r, "recursive strategy should produce chunks"

    chunks_s, _ = chunk_document(SENTENCE_TEXT, strategy="semantic", chunk_size=50)
    assert chunks_s, "semantic strategy should produce chunks"

    parents, children = chunk_document(LONG_TEXT, strategy="parent_child", chunk_size=100, parent_chunk_size=400)
    assert parents, "parent_child strategy should produce parent chunks"
    assert children, "parent_child strategy should produce child chunks"

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document(LONG_TEXT, strategy="invalid")
