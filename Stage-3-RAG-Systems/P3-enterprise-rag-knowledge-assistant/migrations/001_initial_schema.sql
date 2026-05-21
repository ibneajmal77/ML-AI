-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ── documents ─────────────────────────────────────────────────────────────────
-- One row per source document (PDF, DOCX, etc.)
CREATE TABLE IF NOT EXISTS documents (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url   TEXT         NOT NULL,
    file_hash    TEXT         NOT NULL,          -- SHA-256 of file bytes (dedup)
    owner_dept   TEXT         NOT NULL,
    access_level INT          NOT NULL DEFAULT 0, -- 0=public 1=internal 2=confidential 3=restricted
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    tags         TEXT[]       NOT NULL DEFAULT '{}',
    metadata     JSONB        NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash  ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_owner_dept ON documents(owner_dept);

-- ── parent_chunks ─────────────────────────────────────────────────────────────
-- Large context chunks (2048 tokens) — sent to LLM as context
-- Child chunks retrieved first, then parent content fetched for richer context
CREATE TABLE IF NOT EXISTS parent_chunks (
    id             UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content        TEXT  NOT NULL,
    page_number    INT,
    chunk_index    INT   NOT NULL,
    access_level   INT   NOT NULL DEFAULT 0
);

-- ── chunks ────────────────────────────────────────────────────────────────────
-- Small child chunks (512 tokens) — embedded + BM25 indexed for retrieval
-- Points back to its parent_chunk (if parent_child strategy was used)
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_chunk_id UUID         REFERENCES parent_chunks(id) ON DELETE CASCADE,
    content         TEXT         NOT NULL,
    embedding       vector(3072),                -- text-embedding-3-large dims
    embedding_ml    vector(1024),                -- multilingual-e5-large (optional, for GCC)
    page_number     INT,
    chunk_index     INT          NOT NULL,
    token_count     INT          NOT NULL DEFAULT 0,
    access_level    INT          NOT NULL DEFAULT 0,
    owner_dept      TEXT         NOT NULL,
    source_url      TEXT         NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    tags            TEXT[]       NOT NULL DEFAULT '{}'
);

-- HNSW index for fast approximate nearest neighbour search (cosine distance)
-- m=16, ef_construction=64 are standard starting values — tune for recall vs speed
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search column (generated, always up to date)
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS fts_vector TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_fts          ON chunks USING gin(fts_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_access_level ON chunks(access_level);
CREATE INDEX IF NOT EXISTS idx_chunks_owner_dept   ON chunks(owner_dept);

-- ── query_cache ───────────────────────────────────────────────────────────────
-- DB-side backup cache (for Redis miss + persistence across restarts)
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key  TEXT         PRIMARY KEY,
    result     JSONB        NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ  NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_cache_expires ON query_cache(expires_at);
