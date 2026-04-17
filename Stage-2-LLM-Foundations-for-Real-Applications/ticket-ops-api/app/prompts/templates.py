from __future__ import annotations


def classify_user_message(ticket_text: str) -> str:
    return f"""Examples:

Input: "I was charged twice for my subscription this month."
Output: billing

Input: "The dashboard crashes every time I open the Reports tab."
Output: technical

Input: "I need to update the email address on my account."
Output: account

Input: "Just wanted to say the onboarding was smooth."
Output: general

Now classify:
Input: {ticket_text}
Output:"""


def extract_user_message(ticket_text: str) -> str:
    return f"""Extract structured fields from the support ticket below.

Return a JSON object with exactly these fields and no others:
{{
  "issue_type": "<short lowercase label>",
  "urgency": "low" | "medium" | "high" | null,
  "account_id": "<account id>" | null,
  "submitted_at": "YYYY-MM-DD" | null
}}

Rules:
- Return JSON only.
- Use null for missing values.
- Do not infer values that are not explicitly stated.

<ticket>
{ticket_text}
</ticket>
"""


def summarize_user_message(ticket_text: str) -> str:
    return f"""Summarize the support ticket in 2-3 sentences.

<ticket>
{ticket_text}
</ticket>
"""
