from __future__ import annotations

from fastapi import APIRouter

from app.routers.common import validate_ticket_budget
from app.schemas.ticket import ClassifyResponse, TicketRequest
from app.services.classify import classify_ticket


router = APIRouter(tags=["tickets"])


@router.post("/classify", response_model=ClassifyResponse)
@router.post("/tickets/classify", response_model=ClassifyResponse, include_in_schema=False)
def classify(request: TicketRequest) -> ClassifyResponse:
    validate_ticket_budget(request.text)
    return classify_ticket(request.text)
