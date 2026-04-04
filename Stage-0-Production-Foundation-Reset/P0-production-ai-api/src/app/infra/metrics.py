from __future__ import annotations

from collections import Counter
from threading import Lock


class InMemoryMetrics:
    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()
        self._lock = Lock()

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [f"{name} {value}" for name, value in sorted(self._counters.items())]
        return "\n".join(lines) + ("\n" if lines else "")

