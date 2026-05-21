from __future__ import annotations

import hashlib
import json
import logging

from redis.asyncio import Redis

from src.config import settings

logger = logging.getLogger(__name__)


def _make_cache_key(
    query: str,
    user_clearance: int,
    owner_dept: str | None,
    use_hyde: bool = False,
    use_multi_query: bool = False,
    tags: list[str] | None = None,
) -> str:
    tags_str = ",".join(sorted(tags)) if tags else ""
    key_data = f"{query}|{user_clearance}|{owner_dept or ''}|{use_hyde}|{use_multi_query}|{tags_str}"
    digest = hashlib.sha256(key_data.encode()).hexdigest()
    return f"rag:query:{digest}"


class QueryCache:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def get(
        self,
        query: str,
        user_clearance: int,
        owner_dept: str | None,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        tags: list[str] | None = None,
    ) -> dict | None:
        key = _make_cache_key(query, user_clearance, owner_dept, use_hyde, use_multi_query, tags)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(
        self,
        query: str,
        user_clearance: int,
        owner_dept: str | None,
        result: dict,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        key = _make_cache_key(query, user_clearance, owner_dept, use_hyde, use_multi_query, tags)
        await self._redis.setex(key, settings.cache_ttl_seconds, json.dumps(result))

    async def invalidate_pattern(self, pattern: str = "rag:query:*") -> int:
        count = 0
        async for key in self._redis.scan_iter(pattern):
            await self._redis.delete(key)
            count += 1
        return count
