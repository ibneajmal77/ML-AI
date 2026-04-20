import pytest

from app.config import TASK_CONFIGS, get_task_config


def test_get_task_config_classification_is_deterministic() -> None:
    config = get_task_config("classification")
    assert config.temperature == 0.0


def test_get_task_config_drafting_has_variance() -> None:
    config = get_task_config("drafting")
    assert config.temperature >= 0.5


def test_get_task_config_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_task_config("nonexistent_task")


def test_all_task_configs_have_max_tokens_set() -> None:
    for task, config in TASK_CONFIGS.items():
        assert config.max_tokens > 0, f"Task {task!r} has no max_tokens set"
        assert config.max_tokens <= 4096


def test_classification_has_newline_stop_sequence() -> None:
    config = get_task_config("classification")
    assert config.stop == ["\n"]


def test_get_task_config_resolves_model() -> None:
    for task in TASK_CONFIGS:
        config = get_task_config(task)
        assert config.model is not None
