import pytest

from app.prompts.system import SystemPromptBuilder, get_system_prompt


def test_all_task_keys_are_registered() -> None:
    for task in ["classification", "extraction", "summarization", "draft"]:
        prompt = get_system_prompt(task)
        assert isinstance(prompt, str)
        assert len(prompt) > 100


def test_unknown_task_raises_key_error() -> None:
    with pytest.raises(KeyError, match="nonexistent"):
        get_system_prompt("nonexistent")


def test_classify_prompt_output_format_before_constraints() -> None:
    prompt = get_system_prompt("classification")
    assert prompt.index("## Output format") < prompt.index("## Constraints")


def test_classify_prompt_has_example_separator() -> None:
    prompt = get_system_prompt("classification")
    assert "## Examples" in prompt
    assert "\n---\n" in prompt


def test_extract_prompt_contains_required_keys() -> None:
    prompt = get_system_prompt("extraction")
    for key in ["issue_type", "urgency", "account_id", "submitted_at"]:
        assert key in prompt


def test_system_prompt_builder_section_order() -> None:
    prompt = SystemPromptBuilder(
        role="Role.",
        task="Task.",
        output_format="Format.",
        constraints=["Constraint."],
        examples=[("input", "output")],
    ).build()
    assert prompt.index("## Task") < prompt.index("## Output format")
    assert prompt.index("## Output format") < prompt.index("## Constraints")
    assert prompt.index("## Constraints") < prompt.index("## Examples")


def test_system_prompt_loader_reuses_static_string() -> None:
    first = get_system_prompt("classification")
    second = get_system_prompt("classification")
    assert first == second
