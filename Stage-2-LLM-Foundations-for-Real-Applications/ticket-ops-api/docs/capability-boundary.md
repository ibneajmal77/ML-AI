# Internal Copilot - LLM Capability Boundary

## Summary

The LLM is used directly when the task is a language transformation over content
already present in the request. Any task that needs fresh state, private business
data, or exact deterministic execution must be grounded, validated, or handled by
conventional code.

## Tasks The LLM Handles Directly

| Task | Input | Why it is safe |
|---|---|---|
| Classify a ticket | Ticket text | The label comes from the submitted text, not hidden data. |
| Extract fields from a ticket | Ticket text | Extraction is bounded to source text and validated by schema. |
| Summarize a ticket | Ticket text | Summarization rewrites provided content rather than inventing state. |
| Draft an internal support reply | Ticket text plus approved context | The model generates language, not final business actions. |

## Tasks That Require Grounding

| Task | Why grounding is required | Grounding mechanism |
|---|---|---|
| Report a customer's latest payment | Pretraining does not include live billing data. | Tool call or database query |
| Confirm whether a ticket is still open | The answer depends on current system state. | Ticket lookup tool |
| Apply internal routing policy revisions | Internal policies are proprietary and change. | Retrieved policy context |
| Explain whether an outage is active | Requires real-time operational facts. | Incident API or status feed |

## Tasks Outside LLM Scope

| Task | Why |
|---|---|
| Authenticate users | Security-critical logic must be deterministic and auditable. |
| Execute refunds or billing changes | Irreversible actions need explicit code paths and authorization checks. |
| Compute exact SLA compliance | Deterministic rules and timestamps should be handled by code, not generation. |

## Decision Rule

Use the LLM directly only for transformation of supplied text. Add retrieval,
tooling, or validation whenever the answer depends on facts outside the prompt,
must be exact, or would trigger an irreversible downstream action.
