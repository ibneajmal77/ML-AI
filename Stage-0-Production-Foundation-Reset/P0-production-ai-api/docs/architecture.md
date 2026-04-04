# Architecture

## Layered design

```text
api -> service -> domain -> infra
```

- `api`: transport layer, route wiring, schema binding, HTTP mapping
- `service`: orchestration, policy, sync vs async decisions, cost estimation
- `domain`: shared models and application errors
- `infra`: database, provider client, parsing, metrics, logging, auth, rate limiting

## Major design decisions

- Thin routes and thicker services
- Sync summarization for small text
- Async job workflow for larger or multi-step document processing
- SQLite for local practicality, structured so a Postgres repo can replace it later
- Provider abstraction so the local summarizer can be swapped for a real LLM client
- Structured logs with correlation IDs
- Simple in-memory metrics and rate limiting for learning purposes

## Gaps that a real production system would improve

- Replace in-memory metrics with Prometheus/OpenTelemetry
- Replace background tasks with a durable queue
- Replace SQLite with Postgres
- Move auth and rate limiting to gateway or shared middleware
- Add distributed tracing and richer SLO dashboards

