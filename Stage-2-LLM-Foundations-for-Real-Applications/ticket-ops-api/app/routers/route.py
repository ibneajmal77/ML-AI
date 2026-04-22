from __future__ import annotations

from fastapi import APIRouter

from app.routers.common import validate_ticket_budget
from app.schemas.ticket import RouteRequest, RouteResponse
from app.services.route import route_ticket


router = APIRouter(tags=["routing"])


@router.post("/route", response_model=RouteResponse)
@router.post("/tickets/route", response_model=RouteResponse, include_in_schema=False)
def route(request: RouteRequest) -> RouteResponse:
    validate_ticket_budget(request.text)
    return route_ticket(request.text)
