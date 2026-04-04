from __future__ import annotations

import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        now = time.time()
        window_start = now - 60
        bucket = self._events[key]

        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self._limit:
            return False

        bucket.append(now)
        return True

