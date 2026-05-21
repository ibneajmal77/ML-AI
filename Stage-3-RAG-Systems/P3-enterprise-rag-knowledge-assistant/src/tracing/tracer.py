from __future__ import annotations

import logging

from src.config import settings

logger = logging.getLogger(__name__)


class _NullSpan:
    """No-op span returned when Langfuse is not configured."""

    def end(self, **kwargs) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass


def _get_langfuse():
    if not settings.langfuse_public_key:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:
        logger.warning("Langfuse init failed, tracing disabled: %s", exc)
        return None


_langfuse_instance = None


def _get_langfuse_instance():
    global _langfuse_instance
    if _langfuse_instance is None:
        _langfuse_instance = _get_langfuse()
    return _langfuse_instance


class RAGTrace:
    """Wraps a Langfuse trace for a single query request.

    All methods are safe to call regardless of whether Langfuse is configured.
    When disabled, all operations are no-ops via _NullSpan.
    """

    def __init__(self, query: str, user_clearance: int) -> None:
        self._lf = _get_langfuse_instance()
        self._spans: dict[str, object] = {}
        self._trace = None

        if self._lf:
            try:
                self._trace = self._lf.trace(
                    name="rag-query",
                    input={"query": query, "clearance": user_clearance},
                )
            except Exception as exc:
                logger.warning("Failed to create Langfuse trace: %s", exc)
                self._lf = None

    def span(self, name: str, input_data: dict | None = None) -> object:
        if not self._trace:
            span = _NullSpan()
            self._spans[name] = span
            return span
        try:
            span = self._trace.span(name=name, input=input_data)
            self._spans[name] = span
            return span
        except Exception:
            span = _NullSpan()
            self._spans[name] = span
            return span

    def end_span(self, name: str, output: dict | None = None, metadata: dict | None = None) -> None:
        span = self._spans.get(name)
        if span:
            span.end(output=output, metadata=metadata)

    def end(self, output: dict | None = None) -> None:
        if self._trace:
            try:
                self._trace.update(output=output)
            except Exception:
                pass
