from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.deps import get_bm25_index, get_db_pool
from src.api.routes import health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the DB connection pool (sets hnsw.ef_search=100 per connection via init callback)
    await get_db_pool()
    # Seed BM25 index from DB before serving traffic (may take 10-60s for large corpus)
    await get_bm25_index()
    yield
    # Shutdown: pool is GC-collected (acceptable at this scale)


app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
