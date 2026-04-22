def test_route_endpoint_uses_ticket_history_when_context_requires_it(client) -> None:
    response = client.post(
        "/route",
        json={"text": "Account AC-1001 reports the same invoice dispute again for the third time."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assigned_team"] == "billing"
    assert body["used_history"] is True
    assert "history" in body["reasoning"].lower()
    assert body["prompt_version"] == "v1"
    assert body["input_tokens"] > 0
    assert body["output_tokens"] > 0


def test_route_endpoint_can_route_without_tool_use(client) -> None:
    response = client.post(
        "/route",
        json={"text": "The dashboard crashes whenever I export reports."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assigned_team"] == "technical"
    assert body["used_history"] is False
