from __future__ import annotations

from app.config import get_task_config
from app.prompts import classify_v1  # noqa: F401
from app.prompts.registry import latest
from app.schemas.ticket import ClassifyResponse
from app.services.llm import chat_with_retry


VALID_LABELS = {"billing", "technical", "account", "general"}


def classify_ticket(ticket_text: str) -> ClassifyResponse:
    prompt = latest("classify")
    config = get_task_config("classification")
    result = chat_with_retry(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.render_user_message(ticket_text)},
        ],
        config,
    )
    label = result.content.strip().lower()
    if label not in VALID_LABELS:
        label = "general"
    return ClassifyResponse(
        label=label,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        prompt_version=prompt.version,
    )
