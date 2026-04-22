from __future__ import annotations

import json


FAKE_TICKET_DB: dict[str, list[dict]] = {
    "AC-1001": [
        {"id": "T-0091", "date": "2026-04-03", "category": "billing", "resolved": True},
        {"id": "T-0104", "date": "2026-04-18", "category": "billing", "resolved": False},
    ],
    "AC-2002": [
        {"id": "T-0115", "date": "2026-04-15", "category": "technical", "resolved": False},
    ],
    "AC-3003": [],
}


def get_ticket_history(account_id: str) -> list[dict]:
    return FAKE_TICKET_DB.get(account_id, [])


def dispatch_tool_call(name: str, arguments: dict) -> str:
    if name == "get_ticket_history":
        account_id = arguments.get("account_id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("get_ticket_history requires a non-empty account_id")
        return json.dumps(get_ticket_history(account_id))
    raise ValueError(f"Unknown tool: {name}")
