# app/services/llm.py
"""
OpenAI client wrapper.

All service modules call chat_with_retry() — they do not construct the OpenAI
client or set parameters directly. This is the single place where the API client
is constructed, retry logic lives, and per-call token usage is logged.

Retry strategy, caching, and cost optimization are expanded in 2.15.
"""

import logging
import time

from openai import OpenAI, APIError, RateLimitError

from app.config import settings, LLMConfig
from app.utils.tokens import count_messages_tokens

logger = logging.getLogger(__name__)

# ── Client singleton ──────────────────────────────────────────────────────────
# Constructed once at module import. All service calls use this instance.
_client = OpenAI(api_key=settings.openai_api_key)


def chat_with_retry(
    messages: list[dict],
    config: LLMConfig,
    task: str = "unknown",
    retries: int = 2,
) -> str:
    """
    Call the chat completions API with task-specific parameters.

    Returns the assistant message content as a string.
    Retries up to `retries` times on rate limit or transient server errors.
    Logs prompt tokens, completion tokens, and task name on every call.

    Args:
        messages:  OpenAI message format — [{"role": "...", "content": "..."}, ...]
        config:    LLMConfig from get_task_config() — never hardcode params here
        task:      Task name for log correlation
        retries:   Retry attempts on transient errors (default 2)
    """
    prompt_tokens = count_messages_tokens(messages, model=config.model)

    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                stop=config.stop if config.stop else None,
            )

            content = response.choices[0].message.content or ""
            completion_tokens = response.usage.completion_tokens

            logger.info(
                "LLM call complete",
                extra={
                    "task": task,
                    "model": config.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "temperature": config.temperature,
                },
            )

            return content

        except RateLimitError:
            if attempt < retries:
                wait = 2 ** attempt  # 1s then 2s
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{retries + 1}), "
                    f"retrying in {wait}s",
                    extra={"task": task},
                )
                time.sleep(wait)
            else:
                raise

        except APIError as e:
            logger.error(f"API error on task {task!r}: {e}", extra={"task": task})
            raise