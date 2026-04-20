from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from app.config import settings


_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str) -> tiktoken.Encoding:
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODING_CACHE[model] = tiktoken.get_encoding("o200k_base")
    return _ENCODING_CACHE[model]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    if not text:
        return 0
    return len(_get_encoding(model).encode(text))


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    encoder = _get_encoding(model)
    total = 2
    for message in messages:
        total += 4
        for value in message.values():
            total += len(encoder.encode(str(value)))
    return total


def fits_in_budget(text: str, budget: int, model: str = "gpt-4o") -> bool:
    return count_tokens(text, model=model) <= budget


def max_input_tokens(
    context_limit: int | None = None,
    output_reserve: int | None = None,
) -> int:
    if context_limit is None:
        context_limit = settings.context_limit
    if output_reserve is None:
        output_reserve = settings.output_reserve
    return context_limit - output_reserve


def truncate_to_token_budget(
    text: str,
    budget: int,
    model: str = "gpt-4o",
    truncation: str = "tail",
) -> tuple[str, bool]:
    encoder = _get_encoding(model)
    token_ids = encoder.encode(text)
    if len(token_ids) <= budget:
        return text, False
    if truncation == "tail":
        trimmed = token_ids[:budget]
    elif truncation == "middle":
        head = budget // 2
        tail = budget - head
        trimmed = token_ids[:head] + token_ids[-tail:]
    else:
        raise ValueError(f"Unknown truncation strategy: {truncation!r}")
    return encoder.decode(trimmed), True


@dataclass(frozen=True)
class TokenBudget:
    context_limit: int
    output_reserve: int
    system_prompt: int
    instructions: int
    content: int
    tool_results: int = 0
    history: int = 0

    @property
    def input_total(self) -> int:
        return (
            self.system_prompt
            + self.instructions
            + self.content
            + self.tool_results
            + self.history
        )

    @property
    def total(self) -> int:
        return self.input_total + self.output_reserve

    @property
    def content_remaining(self) -> int:
        return self.context_limit - self.total

    def is_valid(self) -> bool:
        return self.total <= self.context_limit

    def summary(self) -> str:
        return (
            f"Budget: system={self.system_prompt} | instr={self.instructions} | "
            f"content={self.content} | tools={self.tool_results} | "
            f"history={self.history} | output={self.output_reserve} | "
            f"total={self.total}/{self.context_limit}"
        )
