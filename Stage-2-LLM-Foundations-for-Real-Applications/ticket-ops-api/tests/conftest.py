from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import ChatResult, ToolCall, set_backend
from app.utils.tokens import count_messages_tokens, count_tokens


class FakeLLMBackend:
    def __init__(self) -> None:
        self.calls = defaultdict(int)

    def chat(
        self,
        messages: list[dict],
        config,
        *,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResult:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        normalized = user.lower()
        self.calls[normalized] += 1

        if "classification system" in system:
            ticket_text = self._extract_classification_text(user)
            label = self._classify(ticket_text.lower())
            return ChatResult(
                content=label,
                input_tokens=count_messages_tokens(messages, config.model or "gpt-4o-mini"),
                output_tokens=count_tokens(label, config.model or "gpt-4o-mini"),
                model=config.model or "gpt-4o-mini",
            )

        if "extraction system" in system:
            ticket_text = self._extract_ticket_text(user)
            payload = self._extract_payload(ticket_text)
            attempt = self.calls[normalized]
            if "broken-json" in ticket_text and attempt == 1:
                content = '{"issue_type": "billing"'
            elif "bad-date" in ticket_text and attempt == 1:
                content = json.dumps(
                    {
                        "issue_type": "billing",
                        "urgency": "medium",
                        "account_id": "AC-1001",
                        "submitted_at": "04/16/2026",
                    }
                )
            else:
                content = json.dumps(payload)
            return ChatResult(
                content=content,
                input_tokens=count_messages_tokens(messages, config.model or "gpt-4o-mini"),
                output_tokens=count_tokens(content, config.model or "gpt-4o-mini"),
                model=config.model or "gpt-4o-mini",
            )

        if "ticket routing system" in system:
            return self._route(messages, config, tools=tools, response_format=response_format)

        raise AssertionError(f"Unexpected prompt family: {system[:80]}")

    def _route(
        self,
        messages: list[dict],
        config,
        *,
        tools: list[dict] | None,
        response_format: dict | None,
    ) -> ChatResult:
        model = config.model or "gpt-4o-mini"
        ticket_text = self._extract_ticket_text(messages[1]["content"])
        last_message = messages[-1]
        if last_message["role"] == "tool":
            history = json.loads(last_message["content"])
            decision = self._route_decision(ticket_text, history=history)
            content = json.dumps(decision)
            return ChatResult(
                content=content,
                input_tokens=count_messages_tokens(messages, model),
                output_tokens=count_tokens(content, model),
                model=model,
                finish_reason="stop",
            )

        account_match = re.search(r"\bAC-\d{4}\b", ticket_text)
        needs_history = account_match is not None and any(
            token in ticket_text.lower() for token in ["third", "again", "repeat", "history"]
        )
        if tools and needs_history:
            tool_call = ToolCall(
                id="tool_call_1",
                name="get_ticket_history",
                arguments=json.dumps({"account_id": account_match.group(0)}),
            )
            return ChatResult(
                content="",
                input_tokens=count_messages_tokens(messages, model),
                output_tokens=count_tokens(tool_call.arguments, model),
                model=model,
                finish_reason="tool_calls",
                tool_calls=[tool_call],
            )

        decision = self._route_decision(ticket_text, history=[])
        content = json.dumps(decision)
        return ChatResult(
            content=content,
            input_tokens=count_messages_tokens(messages, model),
            output_tokens=count_tokens(content, model),
            model=model,
            finish_reason="stop",
        )

    def _extract_ticket_text(self, user_message: str) -> str:
        match = re.search(r"<ticket>\s*(.*?)\s*</ticket>", user_message, re.DOTALL)
        return match.group(1).strip() if match else user_message.strip()

    def _extract_classification_text(self, user_message: str) -> str:
        match = re.search(r"Now classify:\s*Input:\s*(.*?)\s*Output:\s*$", user_message, re.DOTALL)
        return match.group(1).strip() if match else user_message.strip()

    def _classify(self, text: str) -> str:
        if any(token in text for token in ["charged", "invoice", "refund", "billing"]):
            return "billing"
        if any(token in text for token in ["login", "password", "account", "email address"]):
            return "account"
        if any(token in text for token in ["crash", "error", "api", "bug", "dashboard"]):
            return "technical"
        return "general"

    def _extract_payload(self, text: str) -> dict:
        issue_type = self._classify(text.lower())
        urgency = None
        lowered = text.lower()
        if any(token in lowered for token in ["two hours", "urgent", "asap", "board meeting"]):
            urgency = "high"
        elif any(token in lowered for token in ["today", "soon", "blocked", "charged twice"]):
            urgency = "medium"
        elif text.strip():
            urgency = "low"

        account_match = re.search(r"\bAC-\d{4}\b", text)
        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        return {
            "issue_type": issue_type,
            "urgency": urgency,
            "account_id": account_match.group(0) if account_match else None,
            "submitted_at": date_match.group(0) if date_match else None,
        }

    def _route_decision(self, text: str, history: list[dict]) -> dict:
        label = self._classify(text.lower())
        unresolved_count = sum(1 for item in history if not item.get("resolved", True))
        used_history = bool(history)
        reasoning = {
            "billing": "Billing should handle the payment or invoice issue described in the ticket.",
            "technical": "Technical support should handle the product issue described in the ticket.",
            "account": "Account support should handle the access or identity issue described in the ticket.",
            "general": "General support can triage the request because no specialist signal is present.",
        }[label]
        if used_history and unresolved_count:
            suffix = "" if unresolved_count == 1 else "s"
            reasoning = (
                f"{reasoning[:-1]} and account history shows "
                f"{unresolved_count} unresolved related ticket{suffix}."
            )
        return {
            "assigned_team": label,
            "reasoning": reasoning,
            "used_history": used_history,
        }


@pytest.fixture(autouse=True)
def fake_backend() -> Iterator[FakeLLMBackend]:
    backend = FakeLLMBackend()
    set_backend(backend)
    yield backend
    set_backend(None)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
