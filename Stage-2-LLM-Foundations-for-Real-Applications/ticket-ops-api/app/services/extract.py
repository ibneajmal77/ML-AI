from __future__ import annotations

from app.config import get_task_config
from app.prompts import extract_v1  # noqa: F401
from app.prompts.registry import latest
from app.schemas.ticket import ExtractResponse, TicketExtraction
from app.services.llm import chat_json_with_schema


def extract_ticket(ticket_text: str, max_retries: int = 3) -> ExtractResponse:
    prompt = latest("extract")
    config = get_task_config("extraction")
    result, extraction = chat_json_with_schema(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.render_user_message(ticket_text)},
        ],
        config,
        TicketExtraction,
        max_retries=max_retries,
        task="extraction",
    )
    return ExtractResponse(
        data=extraction,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        prompt_version=prompt.version,
    )
