# How endpoints will use this in practice:
from app.config import settings
from app.utils.tokens import count_tokens, truncate_to_token_budget, TokenBudget

budget = TokenBudget(
    context_limit=settings.context_limit,      # 128_000
    output_reserve=settings.output_reserve,    # 2_000
    system_prompt=count_tokens(SYSTEM_PROMPT),
    instructions=count_tokens(INSTRUCTIONS),
    content=0,       # filled per request
    tool_results=0,  # filled per request
)

# Per request:
ticket_tokens = count_tokens(ticket_text)
if ticket_tokens > budget.content:
    ticket_text, truncated = truncate_to_token_budget(ticket_text, budget.content)
    if truncated:
        logger.warning("Ticket text truncated", extra={"original_tokens": ticket_tokens})