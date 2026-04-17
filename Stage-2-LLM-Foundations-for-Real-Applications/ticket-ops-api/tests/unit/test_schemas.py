import pytest
from pydantic import ValidationError

from app.schemas.ticket import TicketExtraction, TicketRequest


def test_ticket_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        TicketRequest(text="")


def test_ticket_extraction_accepts_iso_date() -> None:
    extraction = TicketExtraction(
        issue_type="billing",
        urgency="medium",
        account_id="AC-1001",
        submitted_at="2026-04-16",
    )
    assert extraction.submitted_at == "2026-04-16"


def test_ticket_extraction_rejects_non_iso_date() -> None:
    with pytest.raises(ValidationError):
        TicketExtraction(
            issue_type="billing",
            urgency="medium",
            account_id="AC-1001",
            submitted_at="04/16/2026",
        )
