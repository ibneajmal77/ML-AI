from app.prompts.registry import PromptVersion, register
from app.prompts.system import get_system_prompt
from app.prompts.templates import route_user_message


register(
    PromptVersion(
        name="route",
        version="v1",
        system_prompt=get_system_prompt("routing"),
        render_user_message=route_user_message,
    )
)
