from app.prompts.registry import PromptVersion, register
from app.prompts.system import get_system_prompt
from app.prompts.templates import classify_user_message


register(
    PromptVersion(
        name="classify",
        version="v1",
        system_prompt=get_system_prompt("classification"),
        render_user_message=classify_user_message,
    )
)
