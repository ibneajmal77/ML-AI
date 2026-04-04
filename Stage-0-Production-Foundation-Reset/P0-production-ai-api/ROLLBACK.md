# Rollback

## Goal

Restore the last known-good image and config if a deployment causes user-visible regressions.

## Minimum rollback checklist

1. Identify the current bad release image tag.
2. Identify the previous known-good immutable image tag.
3. Redeploy the previous image without rebuilding.
4. Confirm `/health` and `/ready`.
5. Run the smoke test in `scripts/run_smoke_test.py`.
6. Check latency, 5xx rate, parse failures, and job failure rate.
7. Record the incident and suspected cause.

## Important note

If the regression came from config or prompt changes, rolling back only the container image may not be enough. Restore the matching config version as well.

