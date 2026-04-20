# Internal Copilot - Architecture Notes: Transformer Behavior

Rules derived from transformer behavior. Apply these when designing prompts,
planning context budgets, or debugging inconsistent model behavior.
Last updated: 2026-04-20

## Rule 1 - Place Critical Instructions at the Start, Not the Middle

Source: attention is uneven across long sequences, and content in the middle
receives less reliable attention than content near the beginning or end.

Do:
- Put role, task, and output format at the top of the system prompt.
- Put output format before background context.
- Repeat a short format reminder near the end when a prompt grows long.

Do not:
- Bury output requirements after long background sections.
- Assume that including a rule somewhere in the prompt guarantees compliance.

Applied to the copilot:
- `SystemPromptBuilder` orders sections as `role -> task -> output format -> constraints -> background -> examples`.
- `/classify` and `/extract` keep the output contract above examples and user input.

## Rule 2 - Dense Relevant Context Beats Large Diluted Context

Source: every irrelevant token competes with relevant tokens for attention.

Do:
- Include only the current ticket for first-six-lesson endpoints.
- Delimit user content clearly with `<ticket>...</ticket>`.
- Keep examples short and representative.

Do not:
- Concatenate policy documents, ticket history, or unrelated notes into the first-six-lesson endpoints.
- Use the context window like a storage container for all available data.

Applied to the copilot:
- `/classify` sees only the current ticket and prompt examples.
- `/extract` sees only the current ticket plus the extraction schema contract.

## Rule 3 - Context Window Is a Processing Window, Not a Storage Container

Source: fitting within the model limit is necessary, but it does not guarantee
the model will attend to every token equally well.

Do:
- Measure token usage before live calls.
- Reserve output tokens up front.
- Reject or truncate oversized input before the LLM request path.

Do not:
- Treat `fits in context` as equivalent to `safe and reliable`.
- Let prompt size drift without re-measuring static components.

Applied to the copilot:
- `app/utils/tokens.py` provides `count_tokens()`, `truncate_to_token_budget()`, and `TokenBudget`.
- `POST /classify` and `POST /extract` reject ticket text that exceeds the configured model budget.

## Rule 4 - Domain Vocabulary Must Be Tested Explicitly

Source: embeddings cover common language well, but internal jargon and ID
patterns can drift without targeted examples and tests.

Do:
- Test account IDs, ticket IDs, dates, and category vocabulary explicitly.
- Show representative domain strings in prompt examples.
- Add schema tests for formatting-sensitive fields.

Do not:
- Assume pretraining will always handle internal account code formats correctly.
- Wait for production failures to discover vocabulary blind spots.

Applied to the copilot:
- Prompt examples include category labels, `AC-1001` account IDs, and ISO dates.
- Unit tests validate account ID extraction and date parsing rules.

## Rule 5 - Separate Operator Instructions From User Data

Source: system prompts and user messages are different trust layers. Mixing them
weakens instruction following and increases injection risk.

Do:
- Keep standing instructions in the system prompt.
- Keep runtime ticket text in the user message only.
- Use stable prompt keys so the same system prompt can be loaded repeatedly.

Do not:
- Merge operator rules and user ticket text into one user message.
- Inject per-request data into the system prompt when it belongs in the user turn.

Applied to the copilot:
- Prompt versions store a stable system prompt plus a separate user-message renderer.
- User content is always wrapped in `<ticket>...</ticket>` or explicit input blocks.

## Context Budget Reference Table

Measured from the current prompt artifacts with
`python scripts/measure_prompt_tokens.py` on 2026-04-20.

| Endpoint | System prompt | Instructions/examples | Content | Tool results | Output reserve | Total est. |
|---|---|---|---|---|---|---|
| `/classify` | 158 tok | 88 tok | ~300 tok | - | 20 tok | ~566 tok |
| `/extract` | 179 tok | 125 tok | ~400 tok | - | 300 tok | ~1,004 tok |
| `/summarize` | 57 tok | 35 tok | ~500 tok | - | 180 tok | ~772 tok |
| `/draft` | not measured yet | not measured yet | ~500 tok | - | 400 tok | pending draft endpoint |
| `/chat` | not in scope for lessons 2.1-2.6 | not in scope for lessons 2.1-2.6 | variable | variable | variable | variable |

## Measurement Follow-Through

- Use `scripts/measure_prompt_tokens.py` to replace planning estimates with
  exact counts from the current prompt text.
- Re-run token measurements after any prompt wording change, not just after new
  endpoints are added.
- If prompt growth materially reduces the remaining content budget, update the
  endpoint guardrails in `app/config.py`.
