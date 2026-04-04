from __future__ import annotations

import json
from io import StringIO

import pandas as pd

from app.domain.errors import InvalidContentError


class ContentParser:
    supported_content_types = {"text/plain", "application/json", "text/csv"}

    def parse(self, content: str, content_type: str) -> str:
        if content_type == "text/plain":
            return content.strip()

        if content_type == "application/json":
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise InvalidContentError("Invalid JSON content.") from exc
            return json.dumps(payload, indent=2, sort_keys=True)

        if content_type == "text/csv":
            frame = pd.read_csv(StringIO(content))
            return frame.to_csv(index=False)

        raise InvalidContentError(f"Unsupported content type: {content_type}")

