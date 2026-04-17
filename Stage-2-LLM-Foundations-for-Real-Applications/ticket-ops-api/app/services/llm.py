from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

from openai import APIError, OpenAI, RateLimitError

from app.config import LLMConfig, settings
from app.utils.tokens import count_messages_tokens, count_tokens


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


def chat(messages: list[dict], config: LLMConfig) -> ChatResult:
    return get_backend().chat(messages, config)


def chat_with_retry(
    messages: list[dict],
    config: LLMConfig,
    max_retries: int = 3,
) -> ChatResult:
    attempt = 0
    while True:
        try:
            return chat(messages, config)
        except RateLimitError:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(2 ** (attempt - 1))
        except APIError:
            raise


def parse_json_content(result: ChatResult) -> dict:
    return json.loads(result.content)
