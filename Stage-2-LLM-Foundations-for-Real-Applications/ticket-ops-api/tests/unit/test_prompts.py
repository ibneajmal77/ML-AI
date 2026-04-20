import pytest

from app.prompts import classify_v1, extract_v1  # noqa: F401
from app.prompts.registry import PromptVersion, get, latest, register
from app.prompts.templates import (
    classify_user_message,
    draft_reply_user_message,
    extract_user_message,
    get_template,
)


def test_classify_template_covers_all_categories() -> None:
    message = classify_user_message("test ticket")
    for category in ["billing", "technical", "account", "general"]:
        assert category in message


def test_extract_template_includes_required_keys() -> None:
    message = extract_user_message("ticket")
    for key in ["issue_type", "urgency", "account_id", "submitted_at"]:
        assert key in message
    assert "null" in message


def test_registry_get_returns_registered_prompt() -> None:
    prompt = get("classify", "v1")
    assert prompt.version == "v1"
    assert "classification system" in prompt.system_prompt


def test_registry_missing_prompt_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get("classify", "v99")


def test_registry_latest_prefers_highest_numeric_version() -> None:
    register(
        PromptVersion(
            name="numeric-order-check",
            version="v2",
            system_prompt="v2",
            render_user_message=lambda text: text,
        )
    )
    register(
        PromptVersion(
            name="numeric-order-check",
            version="v10",
            system_prompt="v10",
            render_user_message=lambda text: text,
        )
    )
    assert latest("numeric-order-check").version == "v10"


def test_get_template_raises_for_unknown_task() -> None:
    with pytest.raises(KeyError):
        get_template("nonexistent_task")


def test_draft_reply_includes_context_when_provided() -> None:
    message = draft_reply_user_message(
        "ticket text",
        "billing",
        "high",
        context="Account verified.",
    )
    assert "Account verified." in message


def test_draft_reply_excludes_context_block_when_empty() -> None:
    message = draft_reply_user_message("ticket text", "billing", "high", context="")
    assert "Relevant context" not in message
