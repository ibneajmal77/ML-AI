from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompts import classify_v1, extract_v1, summarize_v1  # noqa: F401
from app.prompts.registry import latest
from app.utils.tokens import count_tokens


def main() -> None:
    prompt_specs = {
        "classify_system_v1": latest("classify").system_prompt,
        "classify_user_example": latest("classify").render_user_message(
            "I was charged twice for my subscription this month."
        ),
        "extract_system_v1": latest("extract").system_prompt,
        "extract_user_example": latest("extract").render_user_message(
            "Account AC-1001 was charged twice on 2026-04-10."
        ),
        "summarize_system_v1": latest("summarize").system_prompt,
        "summarize_user_example": latest("summarize").render_user_message(
            "Customer cannot log in after yesterday's release and needs access before a board meeting."
        ),
    }

    print(f"{'Component':<24} {'Tokens':>6}")
    print("-" * 33)
    for name, text in prompt_specs.items():
        print(f"{name:<24} {count_tokens(text):>6}")


if __name__ == "__main__":
    main()
