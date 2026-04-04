from __future__ import annotations


def test_summarize_requires_api_key(client) -> None:
    response = client.post("/v1/summarize", json={"text": "hello world", "max_sentences": 1})
    assert response.status_code == 401


def test_summarize_success(client) -> None:
    response = client.post(
        "/v1/summarize",
        headers={"x-api-key": "test-key"},
        json={"text": "Sentence one. Sentence two. Sentence three.", "max_sentences": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["processing_mode"] == "sync"
    assert payload["estimated_tokens"] > 0
    assert "Sentence one." in payload["summary"]

