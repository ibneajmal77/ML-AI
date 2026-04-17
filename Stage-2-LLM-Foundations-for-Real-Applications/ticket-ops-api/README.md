# ticket-ops-api

`ticket-ops-api` is a FastAPI project that turns Stage 2 lessons `2.1` through `2.6`
into working artifacts. The repo includes architecture documents, token budgeting
utilities, task-based model configuration, production-style prompt infrastructure,
and live `/classify` and `/extract` endpoints.

## What Is Implemented

- Lesson `2.1`: `docs/capability-boundary.md`
- Lesson `2.2`: `docs/architecture-notes.md`
- Lesson `2.3`: `app/utils/tokens.py`
- Lesson `2.4`: `app/config.py`, `app/services/llm.py`, `/classify`
- Lesson `2.5`: prompt template builders and structured extraction flow
- Lesson `2.6`: system prompt builder, prompt registry, versioned prompts

## Endpoints

- `GET /health`
- `POST /classify`
- `POST /extract`
- `POST /tickets/classify`
- `POST /tickets/extract`

## Run Locally

```bash
pip install -e .[dev]
uvicorn app.main:app --reload
pytest
```

Set `OPENAI_API_KEY` in `.env` if you want live model calls. Tests run with a fake
backend and do not require network access.
