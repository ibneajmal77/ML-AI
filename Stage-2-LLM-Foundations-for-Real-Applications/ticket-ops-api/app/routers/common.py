from __future__ import annotations

from fastapi import HTTPException

from app.config import settings
from app.utils.tokens import fits_in_budget


def validate_ticket_budget(text: str) -> None:
    if not fits_in_budget(text, settings.max_ticket_input_tokens, model=settings.model):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Ticket text exceeds {settings.max_ticket_input_tokens} token limit. "
                "Please shorten the input."
            ),
        )
