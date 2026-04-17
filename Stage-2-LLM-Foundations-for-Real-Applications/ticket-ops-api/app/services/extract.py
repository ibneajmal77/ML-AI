from __future__ import annotations

from pydantic import ValidationError

from app.config import get_task_config
from app.prompts import extract_v1  # noqa: F401
from app.prompts.registry import latest
from app.schemas.ticket import ExtractResponse, TicketExtraction
from app.services.llm import chat_with_retry, parse_json_content


def extract_ticket(ticket_text: str, max_retries: int = 3) -> ExtractResponse:
    prompt = latest("extract")
    config = get_task_config("extraction")
    last_error: Exception | None = None
    for _ in range(max_retries):
        result = chat_with_retry(
            [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.render_user_message(ticket_text)},
            ],
            config,
        )
        try:
            payload = parse_json_content(result)
            extraction = TicketExtraction.model_validate(payload)
            return ExtractResponse(
                data=extraction,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                prompt_version=prompt.version,
            )
        except (ValueError, ValidationError) as exc:
            last_error = exc
    raise last_error or RuntimeError("Extraction failed without an error")
