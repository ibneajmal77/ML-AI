from __future__ import annotations

from app.service.costing import estimate_cost_usd, estimate_tokens


def test_cost_estimation() -> None:
    tokens = estimate_tokens("one two three four")
    cost = estimate_cost_usd(tokens)
    assert tokens > 0
    assert cost > 0

