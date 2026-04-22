from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    strong_model: str = "gpt-4o"
    context_limit: int = 128_000
    output_reserve: int = 2_000
    max_ticket_input_tokens: int = 2_000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


@dataclass(frozen=True)
class LLMConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 200
    stop: list[str] = field(default_factory=list)
    model: str | None = None


TASK_CONFIGS: dict[str, LLMConfig] = {
    "classification": LLMConfig(
        temperature=0.0,
        top_p=1.0,
        max_tokens=20,
        stop=["\n"],
    ),
    "extraction": LLMConfig(
        temperature=0.0,
        top_p=1.0,
        max_tokens=300,
        stop=["}\n\n", "```\n"],
    ),
    "summarization": LLMConfig(
        temperature=0.4,
        top_p=0.9,
        max_tokens=180,
    ),
    "drafting": LLMConfig(
        temperature=0.7,
        top_p=0.95,
        max_tokens=400,
    ),
    "routing": LLMConfig(
        temperature=0.0,
        top_p=1.0,
        max_tokens=220,
    ),
}


def get_task_config(task: str) -> LLMConfig:
    if task not in TASK_CONFIGS:
        raise KeyError(
            f"No config registered for task {task!r}. "
            f"Available: {sorted(TASK_CONFIGS)}"
        )
    config = TASK_CONFIGS[task]
    return LLMConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stop=list(config.stop),
        model=config.model or settings.model,
    )
