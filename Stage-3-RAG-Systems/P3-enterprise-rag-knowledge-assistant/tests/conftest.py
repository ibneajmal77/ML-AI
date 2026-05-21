from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio


def pytest_configure(config):
    """Set required env vars before test collection imports src.config."""
    os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://placeholder.openai.azure.com/")
    os.environ.setdefault("AZURE_OPENAI_API_KEY", "placeholder-key")
    os.environ.setdefault("DATABASE_URL", "postgresql://rag:secret@localhost:5432/ragdb_test")
    os.environ.setdefault("COHERE_API_KEY", "placeholder-key")


from src.domain.models import AccessLevel, Chunk


@pytest_asyncio.fixture(scope="session")
async def pg_pool():
    import asyncpg
    from pathlib import Path
    from src.config import settings

    test_db_url = settings.database_url.replace("/ragdb", "/ragdb_test")

    async def _init(conn):
        await conn.execute("SET hnsw.ef_search = 100")

    pool = await asyncpg.create_pool(test_db_url, min_size=1, max_size=5, init=_init)

    migration_sql = (Path(__file__).parent.parent / "migrations" / "001_initial_schema.sql").read_text()
    await pool.execute(migration_sql)

    yield pool

    await pool.close()


@pytest.fixture
def mock_embeddings():
    return [0.1] * 3072


@pytest.fixture
def sample_chunk():
    return Chunk(
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        document_id=uuid.UUID("87654321-4321-8765-4321-876543218765"),
        content="This is a sample chunk for testing.",
        page_number=1,
        chunk_index=0,
        token_count=8,
        access_level=AccessLevel.INTERNAL,
        owner_dept="engineering",
        source_url="https://example.com/doc.pdf",
        tags=[],
    )
