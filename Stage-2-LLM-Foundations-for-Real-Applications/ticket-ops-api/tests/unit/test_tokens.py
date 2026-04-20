from app.config import settings
from app.utils.tokens import (
    TokenBudget,
    count_tokens,
    fits_in_budget,
    max_input_tokens,
    truncate_to_token_budget,
)


def test_count_tokens_basic() -> None:
    count = count_tokens("Hello, how can I help you today?")
    assert 7 <= count <= 12


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_fits_in_budget_true() -> None:
    assert fits_in_budget("Short text", budget=50) is True


def test_fits_in_budget_false() -> None:
    assert fits_in_budget("word " * 200, budget=50) is False


def test_truncate_returns_shorter() -> None:
    truncated, was_truncated = truncate_to_token_budget("This is a sentence. " * 100, budget=50)
    assert was_truncated is True
    assert count_tokens(truncated) <= 50


def test_truncate_no_change_when_within_budget() -> None:
    result, was_truncated = truncate_to_token_budget("Hello world", budget=100)
    assert was_truncated is False
    assert result == "Hello world"


def test_token_budget_valid() -> None:
    budget = TokenBudget(
        context_limit=128_000,
        output_reserve=2_000,
        system_prompt=600,
        instructions=200,
        content=800,
    )
    assert budget.is_valid() is True
    assert budget.total == 3_600
    assert budget.content_remaining == 124_400


def test_token_budget_invalid() -> None:
    budget = TokenBudget(
        context_limit=1_000,
        output_reserve=500,
        system_prompt=400,
        instructions=300,
        content=200,
    )
    assert budget.is_valid() is False


def test_max_input_tokens_matches_settings_defaults() -> None:
    assert max_input_tokens() == settings.context_limit - settings.output_reserve
