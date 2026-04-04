from __future__ import annotations

import os

import httpx


def main() -> None:
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    api_key = os.getenv("API_KEY", "change-me")
    headers = {"x-api-key": api_key}

    with httpx.Client(timeout=10.0, headers=headers) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()

        ready = client.get(f"{base_url}/ready")
        ready.raise_for_status()

        response = client.post(
            f"{base_url}/v1/summarize",
            json={"text": "This is a smoke test document for the Stage 0 AI API.", "max_sentences": 2},
        )
        response.raise_for_status()
        payload = response.json()
        assert "summary" in payload


if __name__ == "__main__":
    main()

