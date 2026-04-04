from __future__ import annotations

import re


class LocalSummarizerClient:
    def summarize(self, text: str, max_sentences: int) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(sentences[:max_sentences]).strip()

