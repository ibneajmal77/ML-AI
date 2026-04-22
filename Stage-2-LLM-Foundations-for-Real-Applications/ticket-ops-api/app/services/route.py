from __future__ import annotations

import json

from app.config import get_task_config
from app.prompts import route_v1  # noqa: F401
from app.prompts.registry import latest
from app.schemas.ticket import RouteDecision, RouteResponse
from app.services.llm import chat_json_with_schema, chat_with_retry
from app.tools.definitions import ROUTING_TOOLS
from app.tools.handlers import dispatch_tool_call


def route_ticket(ticket_text: str) -> RouteResponse:
    prompt = latest("route")
    config = get_task_config("routing")
    messages = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.render_user_message(ticket_text)},
    ]

    first_result = chat_with_retry(
        messages,
        config,
        task="routing",
        tools=ROUTING_TOOLS,
        tool_choice="auto",
    )
    total_input_tokens = first_result.input_tokens
    total_output_tokens = first_result.output_tokens

    if first_result.finish_reason == "tool_calls":
        messages.append(
            {
                "role": "assistant",
                "content": first_result.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in first_result.tool_calls
                ],
            }
        )
        for tool_call in first_result.tool_calls:
            arguments = json.loads(tool_call.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": dispatch_tool_call(tool_call.name, arguments),
                }
            )
        final_result, decision = chat_json_with_schema(
            messages,
            config,
            RouteDecision,
            task="routing",
        )
        total_input_tokens += final_result.input_tokens
        total_output_tokens += final_result.output_tokens
    elif first_result.finish_reason == "stop":
        decision = RouteDecision.model_validate_json(first_result.content)
    else:
        raise RuntimeError(
            f"Routing ended with unexpected finish_reason={first_result.finish_reason!r}"
        )

    return RouteResponse(
        assigned_team=decision.assigned_team,
        reasoning=decision.reasoning,
        used_history=decision.used_history,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        prompt_version=prompt.version,
    )
