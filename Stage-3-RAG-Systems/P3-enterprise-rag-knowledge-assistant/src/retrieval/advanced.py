from __future__ import annotations

import logging
from functools import lru_cache

from src.domain.models import ScoredChunk
from src.embedding.encoder import get_query_embedding

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import AsyncAzureOpenAI
    from src.config import settings
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


async def hyde_embedding(query: str) -> list[float]:
    """Generate a hypothetical answer to the query and embed it instead.

    HyDE bridges vocabulary mismatch: when query words differ from document
    words, the generated answer uses document-like vocabulary for better recall.
    """
    from src.config import settings

    client = _get_openai_client()
    response = await client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {
                "role": "user",
                "content": f"Write a short factual paragraph that directly answers: {query}",
            }
        ],
        max_tokens=150,
        temperature=0.3,
    )
    hypothetical_answer = response.choices[0].message.content or query
    return await get_query_embedding(hypothetical_answer)


async def generate_query_variations(query: str, n: int = 3) -> list[str]:
    """Generate N rephrased versions of the query to improve recall."""
    from src.config import settings

    client = _get_openai_client()
    try:
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Generate {n} different phrasings of this question that preserve "
                        f"the original meaning. Output only questions, one per line, no numbering: {query}"
                    ),
                }
            ],
            max_tokens=200,
            temperature=0.5,
        )
        content = response.choices[0].message.content or ""
        variations = [line.strip() for line in content.split("\n") if line.strip()]
        return variations[:n] if variations else [query]
    except Exception as exc:
        logger.warning("Query variation generation failed: %s", exc)
        return [query]


def merge_results(result_sets: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    """Deduplicate and merge results from multiple query variations.

    Keeps the highest score for duplicate chunks.
    """
    best: dict[str, ScoredChunk] = {}

    for result_set in result_sets:
        for sc in result_set:
            chunk_id = str(sc.chunk.id)
            if chunk_id not in best or sc.score > best[chunk_id].score:
                best[chunk_id] = sc

    merged = sorted(best.values(), key=lambda x: x.score, reverse=True)
    for i, sc in enumerate(merged):
        sc.rank = i
    return merged
