# Internal Copilot - Architecture Notes

Rules in this file translate transformer behavior into implementation decisions
for `ticket-ops-api`.

## Rule 1 - Put Critical Instructions First

Attention is strongest near the beginning of the prompt. Output requirements and
hard constraints belong at the top of system prompts, not buried after context.

Applied to the copilot:
- Classification and extraction prompts state output format before examples.
- Fallback behavior is defined before user content is injected.

## Rule 2 - Dense Relevant Context Beats Large Context

Adding irrelevant tokens dilutes attention. Only include the current ticket in the
first-six-lesson endpoints; do not attach history or policy documents yet.

Applied to the copilot:
- `/classify` gets only the current ticket.
- `/extract` gets only the current ticket inside `<ticket>` tags.

## Rule 3 - Budget Tokens Explicitly

Context windows are architectural limits, not a convenience setting. Reserve
output tokens and validate input before model calls.

Applied to the copilot:
- `TokenBudget` plans prompt allocations.
- Long ticket payloads are rejected before the LLM call path.

## Rule 4 - Test Domain Vocabulary

Embeddings handle common language well but internal jargon can drift. Category
labels, ticket IDs, and account identifiers must be represented explicitly in
examples and tests.

Applied to the copilot:
- Prompt examples cover all four categories.
- Extraction tests cover `AC-1001`-style account identifiers and ISO dates.

## Rule 5 - Separate Operator Instructions From User Data

System prompts are the operator contract; user messages are untrusted runtime
inputs. Mixing them weakens instruction-following and increases injection risk.

Applied to the copilot:
- `SystemPromptBuilder` creates stable operator prompts.
- User ticket text is always wrapped in `<ticket>...</ticket>`.
