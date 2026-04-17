from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


UserMessageRenderer = Callable[[str], str]


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    system_prompt: str
    render_user_message: UserMessageRenderer


_REGISTRY: dict[tuple[str, str], PromptVersion] = {}


def register(prompt: PromptVersion) -> None:
    _REGISTRY[(prompt.name, prompt.version)] = prompt


def get(name: str, version: str) -> PromptVersion:
    key = (name, version)
    if key not in _REGISTRY:
        raise KeyError(f"No prompt registered for {name!r} version {version!r}")
    return _REGISTRY[key]


def latest(name: str) -> PromptVersion:
    matches = [prompt for (prompt_name, _), prompt in _REGISTRY.items() if prompt_name == name]
    if not matches:
        raise KeyError(f"No prompt registered for {name!r}")
    return sorted(matches, key=lambda prompt: prompt.version)[-1]
