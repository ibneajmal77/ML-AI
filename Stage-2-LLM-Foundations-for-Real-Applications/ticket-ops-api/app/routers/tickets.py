from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas.ticket import ClassifyResponse, ExtractResponse, TicketRequest
from app.services.classify import classify_ticket
from app.services.extract import extract_ticket
from app.utils.tokens import fits_in_budget


router = APIRouter(tags=["tickets"])


def _validate_budget(text: str) -> None:
    if not fits_in_budget(text, settings.max_ticket_input_tokens, model=settings.model):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Ticket text exceeds {settings.max_ticket_input_tokens} token limit. "
                "Please shorten the input."
            ),
        )


@router.post("/classify", response_model=ClassifyResponse)
@router.post("/tickets/classify", response_model=ClassifyResponse, include_in_schema=False)
def classify(request: TicketRequest) -> ClassifyResponse:
    _validate_budget(request.text)
    return classify_ticket(request.text)


@router.post("/extract", response_model=ExtractResponse)
@router.post("/tickets/extract", response_model=ExtractResponse, include_in_schema=False)
def extract(request: TicketRequest) -> ExtractResponse:
    _validate_budget(request.text)
    return extract_ticket(request.text)
