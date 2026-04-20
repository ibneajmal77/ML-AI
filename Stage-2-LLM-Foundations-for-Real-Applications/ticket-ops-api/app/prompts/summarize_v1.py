from app.prompts.registry import PromptVersion, register
from app.prompts.system import get_system_prompt
from app.prompts.templates import summarize_user_message


register(
    PromptVersion(
        name="summarize",
        version="v1",
        system_prompt=get_system_prompt("summarization"),
        render_user_message=summarize_user_message,
    )
)
