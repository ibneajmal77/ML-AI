from __future__ import annotations


GET_TICKET_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_ticket_history",
        "description": (
            "Retrieve recent ticket history for an account. Use this when repeated unresolved "
            "issues or recent account history materially affect the routing decision. "
            "Do not call it for straightforward tickets that can be routed from the current text alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier from the ticket text. Format: AC-1234.",
                }
            },
            "required": ["account_id"],
        },
    },
}


ROUTING_TOOLS = [GET_TICKET_HISTORY_TOOL]
