from __future__ import annotations

from app.config import get_task_config
from app.prompts import summarize_v1  # noqa: F401
from app.prompts.registry import latest
from app.services.llm import chat_with_retry


def summarize_ticket(ticket_text: str) -> str:
    prompt = latest("summarize")
    config = get_task_config("summarization")
    result = chat_with_retry(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.render_user_message(ticket_text)},
        ],
        config,
    )
    return result.content
