# How endpoints will use this in practice:
from app.config import settings
from app.utils.tokens import count_tokens, truncate_to_token_budget, TokenBudget

budget = TokenBudget(
    context_limit=settings.context_limit,      # 128_000
    output_reserve=settings.output_reserve,    # 2_000
    system_prompt=count_tokens(SYSTEM_PROMPT),
    instructions=count_tokens(INSTRUCTIONS),
    content=0,       # filled per request
    tool_results=0,  # filled per request
)

# Per request:
ticket_tokens = count_tokens(ticket_text)
if ticket_tokens > budget.content:
    ticket_text, truncated = truncate_to_token_budget(ticket_text, budget.content)
    if truncated:
        logger.warning("Ticket text truncated", extra={"original_tokens": ticket_tokens})



# app/config.py

from dataclasses import dataclass, field
from pydantic_settings import BaseSettings


# ── Global settings ───────────────────────────────────────────────────────────
# Loaded from environment / .env file once at startup.
# Required: OPENAI_API_KEY

class Settings(BaseSettings):
    """
    Global application configuration.

    Set these in .env or environment variables before running the app.
    """
    openai_api_key: str
    model: str = "gpt-4o-mini"       # Default model for most tasks
    strong_model: str = "gpt-4o"     # Higher quality model for complex tasks (→ 2.12)
    context_limit: int = 128_000     # Max context window in tokens
    output_reserve: int = 2_000      # Default token reserve for output budget planning (→ 2.3)

    class Config:
        env_file = ".env"


settings = Settings()


# ── Per-task parameter configuration ─────────────────────────────────────────
# Each task type has its own behavioral contract.
# Parameters are set per task — not globally.
# Add new task configs to TASK_CONFIGS below; never hardcode parameters in service files.

@dataclass
class LLMConfig:
    """
    Parameter set for one task type.

    Pass this to chat_with_retry() — it unpacks into the API call.
    model=None means use settings.model (resolved in get_task_config).
    """
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 200
    stop: list[str] = field(default_factory=list)
    model: str | None = None  # None → resolved to settings.model at call time


# ── Task configuration registry ───────────────────────────────────────────────
# One entry per task type. Document the rationale for each parameter choice.
# When a parameter changes, change it here — not in every service file that uses it.

TASK_CONFIGS: dict[str, LLMConfig] = {
    "classification": LLMConfig(
        temperature=0.0,    # Deterministic — same ticket must return same label every run
        top_p=1.0,          # Irrelevant at temp=0 but explicit for clarity
        max_tokens=10,      # Output is a single label: "billing", "technical", etc.
        stop=["\n"],        # Stop after the label — prevents few-shot continuation
    ),
    "extraction": LLMConfig(
        temperature=0.0,    # JSON must be structurally stable across runs
        top_p=1.0,
        max_tokens=300,     # JSON object with ~5–7 fields + string values
        stop=[],            # No stop sequences — JSON structure handles termination
    ),
    "summarization": LLMConfig(
        temperature=0.4,    # Some variation is acceptable — summaries can be paraphrased
        top_p=0.9,          # Filter extreme tail; allow natural vocabulary variation
        max_tokens=150,     # 2–3 sentence target
        stop=[],
    ),
    "drafting": LLMConfig(
        temperature=0.7,    # Variation is desirable — different tone each run
        top_p=0.95,
        max_tokens=400,     # 1–2 paragraph reply
        stop=[],
    ),
}


def get_task_config(task: str) -> LLMConfig:
    """
    Return the LLMConfig for a task type with model resolved to settings.model
    if no task-specific model is set.

    Raises KeyError for unregistered tasks — never silently falls back.
    """
    if task not in TASK_CONFIGS:
        raise KeyError(
            f"No config registered for task {task!r}. "
            f"Available: {list(TASK_CONFIGS.keys())}"
        )
    cfg = TASK_CONFIGS[task]
    if cfg.model is None:
        return LLMConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            model=settings.model,
        )
    return cfg