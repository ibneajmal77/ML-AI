# Runbook

## Common incidents

### High latency

- Check `/metrics` for `summarize_requests_total` and latency counters
- Confirm provider delays or oversized payloads
- Check whether clients are sending large content to the sync endpoint
- Shift large requests to the async document endpoint if needed

### Rising job failures

- Check job status counts in the database
- Review structured logs for parser or validation errors
- Confirm content type and file format assumptions

### Ready fails but health passes

- Database path or file permissions may be broken
- Verify startup created the `jobs` table
- Confirm the process can open the configured database path

## Operational rules

- Do not log full user payloads in production
- Do not increase timeouts before checking workload shape
- Do not redeploy from a local manual build
- Prefer rolling back to the last known-good image over ad hoc hotfixing in prod

