from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from openai import APIError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import LLMConfig, settings
from app.utils.tokens import count_messages_tokens, count_tokens


logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_assistant_message(self) -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "arguments": self.arguments,
                    },
                }
            ],
        }


@dataclass(frozen=True)
class ChatResult:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str = "stop"
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMBackend(Protocol):
    def chat(
        self,
        messages: list[dict],
        config: LLMConfig,
        *,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResult: ...


class OpenAIBackend:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for live model calls. "
                "Tests should inject a fake backend before calling the LLM service."
            )
        self._client = OpenAI(api_key=settings.openai_api_key)

    def chat(
        self,
        messages: list[dict],
        config: LLMConfig,
        *,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResult:
        model = config.model or settings.model
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop or None,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in (choice.message.tool_calls or [])
        ]
        usage = response.usage
        completion_text = content or json.dumps(
            [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in tool_calls
            ]
        )
        prompt_tokens = usage.prompt_tokens if usage else count_messages_tokens(messages, model)
        completion_tokens = (
            usage.completion_tokens if usage else count_tokens(completion_text, model)
        )
        return ChatResult(
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            model=model,
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
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
        "llm_call task=%s model=%s prompt_tokens=%s completion_tokens=%s finish_reason=%s tool_calls=%s",
        task or "unknown",
        result.model,
        result.input_tokens,
        result.output_tokens,
        result.finish_reason,
        len(result.tool_calls),
    )


def chat(
    messages: list[dict],
    config: LLMConfig,
    *,
    task: str | None = None,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> ChatResult:
    result = get_backend().chat(
        messages,
        config,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
    )
    _log_usage(result, task)
    return result


def chat_with_retry(
    messages: list[dict],
    config: LLMConfig,
    max_retries: int = 3,
    *,
    task: str | None = None,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> ChatResult:
    attempt = 0
    while True:
        try:
            return chat(
                messages,
                config,
                task=task,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )
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


def chat_json_with_schema(
    messages: list[dict],
    config: LLMConfig,
    schema: type[SchemaT],
    max_retries: int = 3,
    *,
    task: str | None = None,
) -> tuple[ChatResult, SchemaT]:
    last_error: ValidationError | None = None
    for _ in range(max_retries):
        result = chat_with_retry(
            messages,
            config,
            task=task,
            response_format={"type": "json_object"},
        )
        try:
            return result, schema.model_validate_json(result.content)
        except ValidationError as exc:
            last_error = exc
    raise last_error or RuntimeError("JSON validation failed without an error")
