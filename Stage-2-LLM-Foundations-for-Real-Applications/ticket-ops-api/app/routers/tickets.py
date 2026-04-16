# app/routers/tickets.py — /classify endpoint

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.classify import classify_ticket
from app.utils.tokens import count_tokens, fits_in_budget

router = APIRouter(prefix="/tickets", tags=["tickets"])

# Upper bound for a single ticket — anything longer gets rejected before the API call
_MAX_TICKET_INPUT_TOKENS = 2_000


class TicketRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    category: str
    input_tokens: int  # Logged for cost monitoring


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: TicketRequest) -> ClassifyResponse:
    """
    Classify a support ticket into billing, technical, account, or general.

    Returns the category label and input token count for cost monitoring.
    Rejects input over 2,000 tokens with a 422 before making an API call.
    """
    if not fits_in_budget(request.text, _MAX_TICKET_INPUT_TOKENS):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Ticket text exceeds {_MAX_TICKET_INPUT_TOKENS} token limit. "
                "Please shorten the input."
            ),
        )

    category = classify_ticket(request.text)
    return ClassifyResponse(
        category=category,
        input_tokens=count_tokens(request.text),
    )