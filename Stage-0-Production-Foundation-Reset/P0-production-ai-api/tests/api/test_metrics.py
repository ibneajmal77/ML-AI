from __future__ import annotations


def test_metrics_endpoint(client) -> None:
    client.post(
        "/v1/summarize",
        headers={"x-api-key": "test-key"},
        json={"text": "metrics test text", "max_sentences": 1},
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "summarize_requests_total" in response.text

