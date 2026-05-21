from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

logger_name = "src.ingestion.chunker"

_PAGE_PATTERN = re.compile(r'\[Page (\d+)\]')


def _extract_page_number(content: str) -> int | None:
    """Return the last [Page N] marker found in content (inserted by the loader)."""
    matches = _PAGE_PATTERN.findall(content)
    return int(matches[-1]) if matches else None


@dataclass
class RawChunk:
    content: str
    page_number: int | None
    chunk_index: int
    parent_index: int | None = None
    token_count: int = 0


@lru_cache(maxsize=1)
def _get_encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def _split_with_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[RawChunk]:
    if not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    for sep in separators:
        parts = _split_with_separator(text, sep)
        chunks: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for part in parts:
            part_tokens = _count_tokens(part)
            if buffer_tokens + part_tokens + (1 if buffer else 0) > chunk_size:
                if buffer:
                    chunks.append(sep.join(buffer) if sep else "".join(buffer))
                    # Build overlap: take tokens from the end of the flushed chunk
                    overlap_buffer: list[str] = []
                    overlap_tokens = 0
                    for p in reversed(buffer):
                        p_tokens = _count_tokens(p)
                        if overlap_tokens + p_tokens > overlap:
                            break
                        overlap_buffer.insert(0, p)
                        overlap_tokens += p_tokens
                    buffer = overlap_buffer
                    buffer_tokens = overlap_tokens
                buffer.append(part)
                buffer_tokens += part_tokens
            else:
                buffer.append(part)
                buffer_tokens += part_tokens

        if buffer:
            chunks.append(sep.join(buffer) if sep else "".join(buffer))

        # Check if all chunks fit within tolerance
        if all(_count_tokens(c) <= chunk_size * 1.2 for c in chunks):
            return [
                RawChunk(
                    content=c,
                    page_number=_extract_page_number(c),
                    chunk_index=i,
                    token_count=_count_tokens(c),
                )
                for i, c in enumerate(chunks)
                if c.strip()
            ]

    return []


def parent_child_chunk(
    text: str,
    child_size: int = 512,
    parent_size: int = 2048,
    overlap: int = 50,
) -> tuple[list[RawChunk], list[RawChunk]]:
    if not text.strip():
        return [], []

    parents = recursive_chunk(text, chunk_size=parent_size, overlap=overlap)
    children: list[RawChunk] = []

    for parent_idx, parent in enumerate(parents):
        child_raws = recursive_chunk(parent.content, chunk_size=child_size, overlap=overlap)
        for child in child_raws:
            child.parent_index = parent_idx
            child.chunk_index = len(children)
            children.append(child)

    return parents, children


def semantic_chunk(text: str, max_chunk_size: int = 512) -> list[RawChunk]:
    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[RawChunk] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_tokens = _count_tokens(sentence)
        if buffer_tokens + s_tokens > max_chunk_size and buffer:
            joined = " ".join(buffer)
            chunks.append(
                RawChunk(
                    content=joined,
                    page_number=_extract_page_number(joined),
                    chunk_index=len(chunks),
                    token_count=buffer_tokens,
                )
            )
            buffer = []
            buffer_tokens = 0
        buffer.append(sentence)
        buffer_tokens += s_tokens

    if buffer:
        joined = " ".join(buffer)
        chunks.append(
            RawChunk(
                content=joined,
                page_number=_extract_page_number(joined),
                chunk_index=len(chunks),
                token_count=buffer_tokens,
            )
        )

    return chunks


def chunk_document(
    text: str,
    strategy: str,
    chunk_size: int = 512,
    parent_chunk_size: int = 2048,
    overlap: int = 50,
) -> tuple[list[RawChunk], list[RawChunk]]:
    if strategy == "recursive":
        chunks = recursive_chunk(text, chunk_size=chunk_size, overlap=overlap)
        return chunks, chunks
    elif strategy == "parent_child":
        return parent_child_chunk(
            text, child_size=chunk_size, parent_size=parent_chunk_size, overlap=overlap
        )
    elif strategy == "semantic":
        chunks = semantic_chunk(text, max_chunk_size=chunk_size)
        return chunks, chunks
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
