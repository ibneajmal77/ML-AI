from __future__ import annotations

import logging
from functools import lru_cache

from src.config import settings
from src.domain.models import Citation, ScoredChunk
from src.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


@lru_cache(maxsize=1)
def _get_compressor():
    from llmlingua import PromptCompressor
    return PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
    )


def _compress_with_llmlingua(text: str, budget: int) -> str | None:
    try:
        token_count = count_tokens(text)
        target_ratio = max(0.3, budget / max(1, token_count))
        compressor = _get_compressor()
        result = compressor.compress_prompt(text, rate=target_ratio, force_tokens=["\n"])
        return result["compressed_prompt"]
    except Exception as exc:
        logger.warning("LLMLingua compression failed: %s", exc)
        return None


async def assemble_context(
    query: str,
    chunks: list[ScoredChunk],
    vector_store: VectorStore,
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[Citation]]:
    """Build the LLM context string within token budget, with parent swap and compression."""
    history_tokens = 0
    if conversation_history:
        history_text = " ".join(m.get("content", "") for m in conversation_history)
        history_tokens = count_tokens(history_text)

    context_budget = settings.max_context_tokens - min(history_tokens, settings.max_history_tokens)

    context_parts: list[str] = []
    citations: list[Citation] = []
    used_tokens = 0

    for i, sc in enumerate(chunks):
        chunk = sc.chunk
        source_n = i + 1

        # Parent swap: use richer parent content for the LLM context
        if chunk.parent_chunk_id:
            parent_text = await vector_store.get_parent_content(chunk.parent_chunk_id)
            llm_content = parent_text if parent_text else chunk.content
        else:
            llm_content = chunk.content

        chunk_tokens = count_tokens(llm_content)

        if used_tokens + chunk_tokens > context_budget:
            # Try LLMLingua compression before skipping
            remaining = context_budget - used_tokens
            compressed = _compress_with_llmlingua(llm_content, remaining)
            if compressed and count_tokens(compressed) <= remaining:
                llm_content = compressed
                chunk_tokens = count_tokens(llm_content)
            else:
                continue  # Skip this chunk — budget exhausted

        formatted = (
            f"[Source {source_n}: {chunk.source_url}"
            + (f", page {chunk.page_number}" if chunk.page_number else "")
            + f"]\n{llm_content}"
        )
        context_parts.append(formatted)
        used_tokens += chunk_tokens

        citations.append(
            Citation(
                chunk_id=chunk.id,
                source_url=chunk.source_url,
                page_number=chunk.page_number,
                excerpt=chunk.content[:200],
            )
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, citations
