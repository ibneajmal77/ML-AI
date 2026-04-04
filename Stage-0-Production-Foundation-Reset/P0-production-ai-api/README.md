# P0 Production AI API

This is the Stage 0 capstone project for the production foundation lessons. It implements one practical FastAPI service that combines:

- project structure and config management
- request and response validation
- sync and async API patterns
- SQLite-backed job tracking with SQL
- structured logging and simple metrics
- document parsing for text, JSON, and CSV
- basic auth and rate limiting
- cost and token estimation
- tests, Docker, CI, and operational documentation

## Core endpoints

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `POST /v1/summarize`
- `POST /v1/documents/process`
- `GET /v1/documents/jobs/{job_id}`

## Local run

```bash
cp .env.example .env
uv sync --dev
uv run uvicorn app.main:app --reload
```

Call the API with the header:

```text
x-api-key: change-me
```

## Example request

```bash
curl -X POST http://127.0.0.1:8000/v1/summarize \
  -H "content-type: application/json" \
  -H "x-api-key: change-me" \
  -d "{\"text\":\"Summarize this small production AI service example.\",\"max_sentences\":2}"
```

## Lesson coverage

This project maps directly to Stage 0 lessons `0.1` to `0.15`:

- `0.1` repo layout and config
- `0.2` reproducible dependencies with `pyproject.toml`
- `0.3` analytics script using pandas and numpy
- `0.4` text, JSON, and CSV parsing
- `0.5` SQL-backed job persistence
- `0.6` FastAPI routes and dependencies
- `0.7` Pydantic schemas and validation
- `0.8` async job pattern
- `0.9` Docker packaging
- `0.10` logging and metrics
- `0.11` pytest coverage
- `0.12` Azure deployment stub
- `0.13` token and cost estimation
- `0.14` CI workflow and rollback notes
- `0.15` architecture and runbook docs

