def test_classify_endpoint_returns_expected_labels(client) -> None:
    cases = [
        ("I was charged twice for my March invoice.", "billing"),
        ("The dashboard crashes when I open reports.", "technical"),
        ("I cannot reset my password.", "account"),
        ("Just sharing positive feedback on onboarding.", "general"),
    ]
    for text, expected in cases:
        response = client.post("/classify", json={"text": text})
        assert response.status_code == 200
        body = response.json()
        assert body["label"] == expected
        assert body["prompt_version"] == "v1"
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0


def test_classify_endpoint_is_deterministic(client) -> None:
    payload = {"text": "I was charged twice for my subscription this month."}
    labels = [client.post("/classify", json=payload).json()["label"] for _ in range(3)]
    assert labels == ["billing", "billing", "billing"]


def test_classify_endpoint_rejects_empty_input(client) -> None:
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 422
