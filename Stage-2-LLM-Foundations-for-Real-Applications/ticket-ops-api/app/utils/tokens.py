
# app/utils/tokens.py

import tiktoken
from dataclasses import dataclass


# ── Encoding cache ──────────────────────────────────────────────────────────
# tiktoken.encoding_for_model() is fast but creates a new object each call.
# Cache the encoder to avoid repeated initialization.

_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}

def _get_encoding(model: str) -> tiktoken.Encoding:
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for unknown models — use the most recent OpenAI encoding
            _ENCODING_CACHE[model] = tiktoken.get_encoding("o200k_base")
    return _ENCODING_CACHE[model]


# ── Core functions ───────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Return the number of tokens in text for the given model's tokenizer.

    Use this before every API call to verify the prompt fits within budget.
    """
    if not text:
        return 0
    enc = _get_encoding(model)
    return len(enc.encode(text))


def count_messages_tokens(
    messages: list[dict],
    model: str = "gpt-4o",
) -> int:
    """
    Return the token count for a messages list as the API receives it.

    The API adds per-message overhead (~4 tokens per message for role/structure).
    This function accounts for that overhead so counts match what the API charges.
    """
    enc = _get_encoding(model)
    total = 0
    for message in messages:
        total += 4  # per-message overhead: role + structure tokens
        for value in message.values():
            total += len(enc.encode(str(value)))
    total += 2  # reply priming tokens (the model starts its response)
    return total


def fits_in_budget(text: str, budget: int, model: str = "gpt-4o") -> bool:
    """
    Return True if text fits within the given token budget.

    Use to check individual components (user content, tool results)
    before assembling the full prompt.
    """
    return count_tokens(text, model) <= budget


def truncate_to_token_budget(
    text: str,
    budget: int,
    model: str = "gpt-4o",
    truncation: str = "tail",
) -> tuple[str, bool]:
    """
    Truncate text to fit within the given token budget.

    Returns (truncated_text, was_truncated).
    Caller should log when was_truncated=True for monitoring.

    truncation options:
      "tail"   — keep the first `budget` tokens (default, good for ticket text)
      "middle" — keep first budget//2 + last budget//2 tokens
                 (preserves start and end, loses middle — 
                  note: middle content may already get less attention, → 2.2)
    """
    enc = _get_encoding(model)
    token_ids = enc.encode(text)

    if len(token_ids) <= budget:
        return text, False

    if truncation == "tail":
        truncated_ids = token_ids[:budget]
    elif truncation == "middle":
        half = budget // 2
        truncated_ids = token_ids[:half] + token_ids[-half:]
    else:
        raise ValueError(f"Unknown truncation strategy: {truncation!r}")

    return enc.decode(truncated_ids), True


# ── Budget planning ──────────────────────────────────────────────────────────

@dataclass
class TokenBudget:
    """
    Explicit token budget for one API call.

    Plan this before building the prompt — not after.
    Fill each component within its allocation.
    Validate before calling the API.

    Example:
        budget = TokenBudget(
            context_limit=128_000,
            output_reserve=1_000,
            system_prompt=600,
            instructions=200,
            content=800,
            tool_results=0,
        )
        assert budget.is_valid(), f"Budget exceeds context limit: {budget.total}"
    """
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
        """How many tokens are available for additional content beyond planned."""
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