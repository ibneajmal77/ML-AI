# Internal Copilot - LLM Capability Boundary

## Decision Date

2026-04-20

## Summary

This document defines which tasks `ticket-ops-api` lets the LLM handle directly
and which tasks require retrieval, tools, or deterministic code. The rule is
simple: use the model directly only for language transformation of content that
is already present in the prompt.

## Tasks The LLM Handles Directly

These tasks are safe because they operate on supplied text instead of hidden or
live system state.

| Task | Input | Why it is safe |
|---|---|---|
| Classify a ticket into `billing`, `technical`, `account`, or `general` | Ticket text | The output is a label derived from the provided text. It does not require external data beyond what the user submitted. |
| Extract structured fields from a ticket | Ticket text | Extraction is bounded to source text. The service validates the output against a schema before returning it. |
| Summarize a ticket for an internal agent | Ticket text | Summarization is a rewrite of provided content, not a claim about live facts. |
| Draft an internal reply suggestion | Ticket text plus approved context | The model generates language only. A human or deterministic service still owns final action. |

## Tasks That Require Grounding

These tasks require facts the model cannot know from pretraining or the current
prompt alone.

| Task | Why grounding is required | Grounding mechanism |
|---|---|---|
| Report a customer's latest payment | Pretraining gives language patterns, not live billing data. A fluent answer here could be fabricated. | Database query or billing tool |
| Confirm whether a ticket is currently open | Current ticket state is outside the prompt and may change minute to minute. | Ticket lookup tool |
| Apply the latest internal routing policy | Internal policy text is proprietary and can change after model training. | Retrieved policy context |
| Explain whether an outage is still active | Real-time incident state is not available from model pretraining. | Status API or incident feed |

## Tasks Outside LLM Scope

These tasks should not rely on generation even if some supporting context is
available.

| Task | Why |
|---|---|
| Authenticate users | Authentication must be deterministic, auditable, and enforced by conventional security controls. |
| Execute refunds or billing changes | These are irreversible business actions that require explicit authorization and exact logic. |
| Compute exact SLA compliance | Precise rules, arithmetic, and timestamps belong in deterministic code paths, not probabilistic generation. |

## Decision Rule

Use the LLM directly when the task is a transformation of supplied text and the
answer does not need live or hidden facts. Add retrieval, tool calls, or schema
validation when the answer depends on information outside the prompt, must be
factually exact, or could trigger an irreversible downstream action.

## Why This Boundary Exists

- Tokens are the unit of cost and context, not a source of truth. More tokens in
  a prompt do not give the model access to hidden system state.
- Next-token prediction produces probable continuations, not verified facts. A
  fluent unsupported answer is still unsupported.
- Pretraining gives broad language capability and background knowledge, but it
  does not provide access to current private business data.
- Instruction tuning makes the model cooperative and well-formatted. It does
  not make unsupported claims safe to trust.

## Review Trigger

Update this document before adding a new endpoint, tool, or automation path to
`ticket-ops-api`.
