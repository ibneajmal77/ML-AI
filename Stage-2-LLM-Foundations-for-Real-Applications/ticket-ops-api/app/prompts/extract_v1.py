from app.prompts.registry import PromptVersion, register
from app.prompts.system import get_system_prompt
from app.prompts.templates import extract_user_message


register(
    PromptVersion(
        name="extract",
        version="v1",
        system_prompt=get_system_prompt("extract"),
        render_user_message=extract_user_message,
    )
)
