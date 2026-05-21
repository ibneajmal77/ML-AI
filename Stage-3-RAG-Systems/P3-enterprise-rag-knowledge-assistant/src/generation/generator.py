from __future__ import annotations

import logging
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.domain.models import Citation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise knowledge assistant. Answer questions using ONLY the provided context.
Rules:
- If the answer is not found in the context, respond: "I could not find this information in the available documents."
- Always cite your sources using [Source N] references.
- Be factual, concise, and direct.
- Never fabricate information not present in the context."""

_NOT_FOUND_RESPONSE = "I could not find this information in the available documents."


@lru_cache(maxsize=1)
def _get_client():
    from openai import AsyncAzureOpenAI
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _call_llm(messages: list[dict]) -> str:
    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=messages,
        max_tokens=settings.max_response_tokens,
        temperature=0.1,
    )
    content = response.choices[0].message.content
    return content if content else _NOT_FOUND_RESPONSE


async def generate_answer(
    query: str,
    context: str,
    citations: list[Citation],
    conversation_history: list[dict] | None = None,
) -> str:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history[-4:])

    messages.append(
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer (cite sources as [Source N]):"
            ),
        }
    )

    return await _call_llm(messages)
