from __future__ import annotations

from fastapi import APIRouter

from app.routers.common import validate_ticket_budget
from app.schemas.ticket import ExtractResponse, TicketRequest
from app.services.extract import extract_ticket


router = APIRouter(tags=["tickets"])


@router.post("/extract", response_model=ExtractResponse)
@router.post("/tickets/extract", response_model=ExtractResponse, include_in_schema=False)
def extract(request: TicketRequest) -> ExtractResponse:
    validate_ticket_budget(request.text)
    return extract_ticket(request.text)
