import json

import pytest

from app.tools.definitions import GET_TICKET_HISTORY_TOOL
from app.tools.handlers import dispatch_tool_call, get_ticket_history


def test_get_ticket_history_returns_expected_account_data() -> None:
    history = get_ticket_history("AC-1001")
    assert len(history) == 2
    assert history[0]["category"] == "billing"


def test_get_ticket_history_returns_empty_list_for_unknown_account() -> None:
    assert get_ticket_history("AC-UNKNOWN") == []


def test_dispatch_tool_call_returns_json_string() -> None:
    payload = dispatch_tool_call("get_ticket_history", {"account_id": "AC-1001"})
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    assert parsed[0]["id"] == "T-0091"


def test_dispatch_tool_call_raises_for_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        dispatch_tool_call("unknown_tool", {})


def test_ticket_history_tool_schema_has_required_account_id() -> None:
    params = GET_TICKET_HISTORY_TOOL["function"]["parameters"]
    assert params["required"] == ["account_id"]
