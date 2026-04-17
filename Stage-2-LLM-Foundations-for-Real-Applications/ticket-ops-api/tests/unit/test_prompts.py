import pytest

from app.prompts import classify_v1, extract_v1  # noqa: F401
from app.prompts.registry import get, latest
from app.prompts.system import SystemPromptBuilder, get_system_prompt
from app.prompts.templates import classify_user_message, extract_user_message


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


def test_registry_latest_returns_v1() -> None:
    assert latest("extract").version == "v1"


def test_system_prompt_builder_preserves_section_order() -> None:
    prompt = SystemPromptBuilder(
        role="Role.",
        task="Task.",
        output_format="Format.",
        constraints=["Constraint."],
        examples=[("input", "output")],
    ).build()
    assert prompt.index("Role.") < prompt.index("## Task") < prompt.index("## Output format")
    assert prompt.index("## Output format") < prompt.index("## Constraints") < prompt.index("## Examples")


def test_system_prompt_loader_reuses_static_string() -> None:
    first = get_system_prompt("classify")
    second = get_system_prompt("classify")
    assert first == second
