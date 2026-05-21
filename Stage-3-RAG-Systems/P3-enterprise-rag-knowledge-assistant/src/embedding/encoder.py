from __future__ import annotations

import logging
from functools import lru_cache

from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)

_AZURE_MODEL = "text-embedding-3-large"
_ML_MODEL = "intfloat/multilingual-e5-large"


@lru_cache(maxsize=1)
def _get_azure_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


@lru_cache(maxsize=1)
def _get_ml_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_ML_MODEL)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _embed_azure_batch(texts: list[str]) -> list[list[float]]:
    client = _get_azure_client()
    response = await client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=texts,
        dimensions=settings.embedding_dims,
    )
    return [item.embedding for item in response.data]


def _embed_ml_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    model = _get_ml_model()
    prefix = "query: " if is_query else "passage: "
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


async def get_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-large",
) -> list[list[float]]:
    """Embed a list of document texts, batching 100 at a time."""
    if not texts:
        return []

    results: list[list[float]] = []

    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        if model == _ML_MODEL:
            results.extend(_embed_ml_batch(batch, is_query=False))
        else:
            results.extend(await _embed_azure_batch(batch))

    return results


async def get_query_embedding(
    query: str,
    model: str = "text-embedding-3-large",
) -> list[float]:
    """Embed a single query string."""
    if model == _ML_MODEL:
        return _embed_ml_batch([query], is_query=True)[0]
    result = await get_embeddings([query], model=model)
    return result[0]
