from __future__ import annotations


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 2)


def estimate_cost_usd(estimated_tokens: int, rate_per_thousand_tokens: float = 0.0015) -> float:
    return round((estimated_tokens / 1000) * rate_per_thousand_tokens, 6)

