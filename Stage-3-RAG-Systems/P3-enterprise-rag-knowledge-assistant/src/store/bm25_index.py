from __future__ import annotations

import asyncio
import logging

import asyncpg
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._lock = asyncio.Lock()

    async def add_documents(self, doc_ids: list[str], texts: list[str]) -> None:
        tokenized = [text.lower().split() for text in texts]
        async with self._lock:
            self._doc_ids.extend(doc_ids)
            self._corpus.extend(tokenized)
            self._bm25 = BM25Okapi(self._corpus)
        logger.debug("BM25 index rebuilt: %d total documents", len(self._doc_ids))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    async def build_from_db(self, pool: asyncpg.Pool) -> None:
        rows = await pool.fetch("SELECT id::text, content FROM chunks ORDER BY created_at")
        if not rows:
            logger.info("BM25 index: no chunks in DB, starting empty")
            return
        doc_ids = [row["id"] for row in rows]
        texts = [row["content"] for row in rows]
        await self.add_documents(doc_ids, texts)
        logger.info("BM25 index seeded with %d chunks from DB", len(doc_ids))
