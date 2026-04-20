from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from openai import APIError, OpenAI, RateLimitError

from app.config import LLMConfig, settings
from app.utils.tokens import count_messages_tokens, count_tokens


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMBackend(Protocol):
    def chat(self, messages: list[dict], config: LLMConfig) -> ChatResult: ...


class OpenAIBackend:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for live model calls. "
                "Tests should inject a fake backend before calling the LLM service."
            )
        self._client = OpenAI(api_key=settings.openai_api_key)

    def chat(self, messages: list[dict], config: LLMConfig) -> ChatResult:
        model = config.model or settings.model
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop or None,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else count_messages_tokens(messages, model)
        completion_tokens = usage.completion_tokens if usage else count_tokens(content, model)
        return ChatResult(
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            model=model,
        )


_backend: LLMBackend | None = None


def get_backend() -> LLMBackend:
    global _backend
    if _backend is None:
        _backend = OpenAIBackend()
    return _backend


def set_backend(backend: LLMBackend | None) -> None:
    global _backend
    _backend = backend


def _log_usage(result: ChatResult, task: str | None) -> None:
    logger.info(
        "llm_call task=%s model=%s prompt_tokens=%s completion_tokens=%s",
        task or "unknown",
        result.model,
        result.input_tokens,
        result.output_tokens,
    )


def chat(messages: list[dict], config: LLMConfig, *, task: str | None = None) -> ChatResult:
    result = get_backend().chat(messages, config)
    _log_usage(result, task)
    return result


def chat_with_retry(
    messages: list[dict],
    config: LLMConfig,
    max_retries: int = 3,
    *,
    task: str | None = None,
) -> ChatResult:
    attempt = 0
    while True:
        try:
            return chat(messages, config, task=task)
        except RateLimitError as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            logger.warning(
                "rate_limit_retry task=%s attempt=%s error=%s",
                task or "unknown",
                attempt,
                exc.__class__.__name__,
            )
            time.sleep(2 ** (attempt - 1))
        except APIError:
            raise


def parse_json_content(result: ChatResult) -> dict:
    return json.loads(result.content)
