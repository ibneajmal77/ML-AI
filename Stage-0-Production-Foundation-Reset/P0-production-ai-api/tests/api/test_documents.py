from __future__ import annotations


def test_document_job_submission_and_status(client) -> None:
    submit_response = client.post(
        "/v1/documents/process",
        headers={"x-api-key": "test-key"},
        json={
            "content": "a,b\n1,2\n3,4\n",
            "content_type": "text/csv",
            "max_sentences": 2,
        },
    )
    assert submit_response.status_code == 202
    job_id = submit_response.json()["job_id"]

    status_response = client.get(
        f"/v1/documents/jobs/{job_id}",
        headers={"x-api-key": "test-key"},
    )
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] in {"completed", "queued", "running"}
    assert payload["estimated_tokens"] > 0

