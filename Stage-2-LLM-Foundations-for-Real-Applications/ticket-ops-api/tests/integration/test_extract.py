def test_extract_endpoint_returns_all_fields(client) -> None:
    response = client.post(
        "/extract",
        json={
            "text": "Account AC-1001 was charged twice and submitted on 2026-04-10."
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["issue_type"] == "billing"
    assert body["data"]["urgency"] == "medium"
    assert body["data"]["account_id"] == "AC-1001"
    assert body["data"]["submitted_at"] == "2026-04-10"
    assert body["prompt_version"] == "v1"


def test_extract_endpoint_returns_null_for_missing_account_id(client) -> None:
    response = client.post(
        "/extract",
        json={"text": "The customer asked about an invoice discrepancy on 2026-04-12."},
    )
    assert response.status_code == 200
    assert response.json()["data"]["account_id"] is None


def test_extract_endpoint_detects_high_urgency(client) -> None:
    response = client.post(
        "/extract",
        json={"text": "Board meeting in two hours and account AC-1001 still cannot access billing."},
    )
    assert response.status_code == 200
    assert response.json()["data"]["urgency"] == "high"


def test_extract_endpoint_retries_after_bad_json(client) -> None:
    response = client.post(
        "/extract",
        json={"text": "broken-json Account AC-1001 was charged twice on 2026-04-16."},
    )
    assert response.status_code == 200
    assert response.json()["data"]["account_id"] == "AC-1001"


def test_extract_endpoint_retries_after_schema_error(client) -> None:
    response = client.post(
        "/extract",
        json={"text": "bad-date Account AC-1001 was charged twice on 2026-04-16."},
    )
    assert response.status_code == 200
    assert response.json()["data"]["submitted_at"] == "2026-04-16"
