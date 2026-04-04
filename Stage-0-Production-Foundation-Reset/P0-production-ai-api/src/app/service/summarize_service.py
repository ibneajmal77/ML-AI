from __future__ import annotations

from app.api.schemas.summarize import SummarizeRequest
from app.config import Settings
from app.infra.metrics import InMemoryMetrics
from app.infra.provider import LocalSummarizerClient
from app.service.costing import estimate_cost_usd, estimate_tokens


class SummarizeService:
    def __init__(
        self,
        *,
        provider: LocalSummarizerClient,
        metrics: InMemoryMetrics,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._metrics = metrics
        self._settings = settings

    async def summarize(self, request: SummarizeRequest) -> dict:
        normalized_text = request.text.strip()
        estimated_tokens = estimate_tokens(normalized_text)
        estimated_cost_usd = estimate_cost_usd(estimated_tokens)
        summary = self._provider.summarize(normalized_text, request.max_sentences)
        self._metrics.increment("summarize_requests_total")
        return {
            "summary": summary,
            "model_name": self._settings.default_model_name,
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "processing_mode": "sync",
        }

