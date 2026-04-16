# app/services/classify.py — minimal version
# Prompt is upgraded in 2.5 (few-shot examples + constraints from templates.py)

import logging

from app.config import get_task_config
from app.services.llm import chat_with_retry

logger = logging.getLogger(__name__)

# Minimal system prompt — upgraded to production quality in 2.5
CLASSIFY_SYSTEM_PROMPT = (
    "You are a support ticket classification assistant. "
    "Classify support tickets into exactly one of these categories: "
    "billing, technical, account, general. "
    "Return the category label only. No explanation."
)

_VALID_LABELS = {"billing", "technical", "account", "general"}


def classify_ticket(ticket_text: str) -> str:
    """
    Classify a support ticket into one of four categories.

    Returns the category label as a lowercase string.
    Falls back to "general" and logs a warning on unexpected model output.
    Prompt is upgraded with few-shot examples and constraints in 2.5.
    """
    config = get_task_config("classification")
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ]
    raw = chat_with_retry(messages, config, task="classification")
    label = raw.strip().lower()

    if label not in _VALID_LABELS:
        logger.warning(
            "Unexpected classification label — falling back to 'general'",
            extra={"raw_output": label[:100]},
        )
        return "general"

    return label