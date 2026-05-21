# PROJECT SPEC: Enterprise RAG Knowledge Assistant
### Stage 3 — RAG Systems | Portfolio Project P3

> **Purpose of this document:** Complete implementation blueprint.
> An AI coding tool reading this document should be able to build the entire project
> from scratch without needing any additional design decisions.

---

## 1. WHAT THIS PROJECT IS

An enterprise-grade knowledge assistant that lets employees ask questions in natural language
and get grounded answers with citations — sourced from internal documents (PDFs, DOCX, HTML,
PPTX, Markdown).

**Used by in the real world:** Notion AI, Confluence AI, Workday, ServiceNow, law firms
(Allen & Overy), banks (HSBC internal knowledge), consultancies, healthcare knowledge bases.

**Latency SLAs to meet:**
- P95 < 3s with reranker (standard path)
- P95 < 1.5s without reranker (cached queries)

---

## 2. ARCHITECTURE: THE COMPLETE SYSTEM

Two separate flows run in this system:

### FLOW A — INGESTION (document goes in)

```
Upload file (PDF/DOCX/HTML/PPTX/MD)
        │
        ▼
   [Loader] — unstructured.io (default) or Azure Document Intelligence (complex layouts)
        │  Outputs: raw text + page numbers
        ▼
   [Chunker] — 3 strategies (chosen per document type):
        │   recursive     → general docs (default)
        │   parent_child  → technical docs (small child for retrieval, large parent for LLM)
        │   semantic      → legal/medical (split at sentence boundaries)
        ▼
   [Embedder] — text-embedding-3-large (3072 dims) via Azure OpenAI
        │   OR multilingual-e5-large for Arabic+English (GCC deployments)
        ▼
   [Store] — write to PostgreSQL + pgvector:
        │   documents table     → document metadata + file_hash (for dedup)
        │   parent_chunks table → large context chunks
        │   chunks table        → small child chunks + their vector embeddings
        ▼
   [BM25 Index] — add new chunk texts to in-memory BM25 index (rank_bm25)
        ▼
   DONE — document is searchable
```

### FLOW B — QUERY (user asks a question)

```
User sends: { query, user_clearance, owner_dept, use_hyde, use_multi_query }
        │
        ▼
   [Cache Check] — Redis: hash(query + clearance + dept) → if HIT return instantly
        │  (covers 38% of traffic, saves 350ms + Cohere cost)
        │
        ▼ (cache miss)
   [Embed Query] — same embedding model as ingestion
        │  If use_hyde=true: generate hypothetical answer first, embed that instead
        │  If use_multi_query=true: generate 3 query variations, embed all of them
        │
        ▼
   [Hybrid Search] — run BOTH in parallel:
        │   Dense:  pgvector ANN search (cosine similarity) WHERE access_level <= user_clearance
        │   BM25:   keyword search on in-memory BM25 index
        │   Merge:  RRF (Reciprocal Rank Fusion) → top-20 candidates
        │
        ▼
   [Reranker] — Cohere Rerank v3 cross-encoder: top-20 → top-5
        │   (+350ms, improves faithfulness 78% → 91%)
        │
        ▼
   [Context Assembly] — build LLM prompt within token budget:
        │   Budget: 2000 context tokens + 500 history + 200 response max
        │   If parent-child chunking was used: swap child content for parent content
        │   If over budget: compress with LLMLingua before skipping a chunk
        │
        ▼
   [Generate] — Azure OpenAI GPT-4o
        │   System prompt: "answer only from context, say not found if not in context"
        │   Returns: answer text + [Source N] citation markers
        │
        ▼
   [Cache Write] — store result in Redis (TTL: 1 hour)
        │
        ▼
   Return: { answer, citations: [{source_url, page, excerpt}], latency_ms, cache_hit }
```

---

## 3. COMPLETE DIRECTORY STRUCTURE

Every file is listed. No file should be omitted or renamed.

```
Stage-3-RAG-Systems/
└── P3-enterprise-rag-knowledge-assistant/
    │
    ├── pyproject.toml                  ← Python project config + all dependencies
    ├── Dockerfile                      ← Production container (python:3.11-slim)
    ├── docker-compose.yml              ← Full stack: api + worker + beat + postgres + redis
    ├── docker-compose.dev.yml          ← Dev override: volume mounts + hot reload
    ├── .env.example                    ← All required env vars with placeholder values
    ├── .gitignore
    │
    ├── .github/
    │   └── workflows/
    │       ├── ci.yml                  ← Lint + unit tests on every push
    │       └── ragas-eval-gate.yml     ← RAGAS evaluation gate: blocks PR if regression
    │
    ├── migrations/
    │   └── 001_initial_schema.sql      ← Full DB schema (run on container start)
    │
    ├── infra/                          ← Azure deployment (Bicep templates)
    │   ├── container-app.bicep         ← Azure Container Apps definition
    │   └── postgres.bicep              ← Azure Database for PostgreSQL Flexible Server
    │
    ├── src/
    │   ├── __init__.py
    │   ├── config.py                   ← All settings via pydantic-settings (reads .env)
    │   │
    │   ├── domain/
    │   │   ├── __init__.py
    │   │   ├── models.py               ← All Pydantic models (Document, Chunk, QueryRequest, etc.)
    │   │   └── errors.py               ← Custom exception classes
    │   │
    │   ├── ingestion/
    │   │   ├── __init__.py
    │   │   ├── loader.py               ← unstructured.io + Azure Document Intelligence loaders
    │   │   ├── chunker.py              ← 3 chunking strategies: recursive, parent_child, semantic
    │   │   └── pipeline.py             ← Orchestrates: load → chunk → embed → store
    │   │
    │   ├── embedding/
    │   │   ├── __init__.py
    │   │   └── encoder.py              ← text-embedding-3-large + multilingual-e5-large
    │   │
    │   ├── store/
    │   │   ├── __init__.py
    │   │   ├── vector_store.py         ← pgvector: upsert + ANN search with RLS access control
    │   │   └── bm25_index.py           ← In-memory BM25 index (rank_bm25), seeded from DB on startup
    │   │
    │   ├── retrieval/
    │   │   ├── __init__.py
    │   │   ├── hybrid.py               ← RRF: merge dense + BM25 rankings
    │   │   ├── reranker.py             ← Cohere Rerank v3: top-20 → top-5
    │   │   └── advanced.py             ← HyDE embedding + multi-query generation + result merger
    │   │
    │   ├── context/
    │   │   ├── __init__.py
    │   │   └── assembler.py            ← Token budget enforcement + LLMLingua compression
    │   │
    │   ├── generation/
    │   │   ├── __init__.py
    │   │   └── generator.py            ← Azure OpenAI GPT-4o grounded answer synthesis
    │   │
    │   ├── cache/
    │   │   ├── __init__.py
    │   │   └── query_cache.py          ← Redis: cache full query responses, TTL 1 hour
    │   │
    │   ├── indexing/
    │   │   ├── __init__.py
    │   │   ├── worker.py               ← Celery app factory + beat schedules
    │   │   └── tasks.py                ← Tasks: ingest_document, incremental_sync, full_reindex
    │   │
    │   └── tracing/
    │       ├── __init__.py
    │       └── tracer.py               ← Langfuse: full span tree per query
    │
    ├── eval/
    │   ├── __init__.py
    │   ├── golden_dataset.json         ← 200 Q&A pairs for evaluation (see format below)
    │   ├── ragas_config.py             ← RAGAS metric thresholds
    │   └── run_eval.py                 ← Runner: evaluate + exit(1) on regression
    │
    └── tests/
        ├── conftest.py                 ← Shared fixtures: test DB, mock embeddings
        ├── unit/
        │   ├── __init__.py
        │   ├── test_chunker.py         ← Test all 3 chunking strategies
        │   ├── test_hybrid.py          ← Test RRF logic in isolation
        │   └── test_reranker.py        ← Test reranker fallback behavior
        └── integration/
            ├── __init__.py
            ├── test_ingest.py          ← Full ingest flow against test DB
            └── test_query.py           ← Full query flow against test DB
```

---

## 4. ALL DEPENDENCIES (pyproject.toml)

### Runtime dependencies — exact list

```toml
[project]
name = "enterprise-rag-knowledge-assistant"
version = "1.0.0"
requires-python = ">=3.11"

dependencies = [
    # API
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.18",
    # Config + validation
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    # Database
    "asyncpg>=0.30.0",
    "pgvector>=0.3.5",
    # LLM + embeddings
    "openai>=1.56.0",
    "tiktoken>=0.8.0",
    # Retrieval
    "cohere>=5.11.0",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=3.3.0",
    # Document ingestion
    "unstructured[all-docs]>=0.16.0",
    # Context compression
    "llmlingua>=0.2.2",
    # Caching + workers
    "redis>=5.2.0",
    "celery[redis]>=5.4.0",
    # Tracing
    "langfuse>=2.53.0",
    # HTTP + resilience
    "httpx>=0.28.0",
    "tenacity>=9.0.0",
    # Azure Document Intelligence
    "azure-ai-formrecognizer>=3.3.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.8.0",
    "ragas>=0.2.5",
    "datasets>=2.21.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## 5. ENVIRONMENT VARIABLES (.env.example)

Every variable the application reads. All must be present — no optional ones are silently ignored.

```env
# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2024-10-21

# ── PostgreSQL + pgvector ──────────────────────────────────────────────────────
DATABASE_URL=postgresql://rag:secret@localhost:5432/ragdb

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0

# ── Cohere ────────────────────────────────────────────────────────────────────
COHERE_API_KEY=your-cohere-key

# ── Langfuse (tracing) ────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# ── Azure Document Intelligence (optional — for complex PDFs with tables/forms)
AZURE_DI_ENDPOINT=https://your-di.cognitiveservices.azure.com/
AZURE_DI_KEY=your-di-key

# ── Celery workers ────────────────────────────────────────────────────────────
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# ── App behaviour ─────────────────────────────────────────────────────────────
APP_ENV=development
LOG_LEVEL=INFO

# ── Embedding config ──────────────────────────────────────────────────────────
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMS=3072

# ── Retrieval config ──────────────────────────────────────────────────────────
RERANKER_TOP_K=20          # how many candidates to send to Cohere reranker
FINAL_TOP_K=5              # how many to keep after reranking

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE=512             # child chunk size in tokens
PARENT_CHUNK_SIZE=2048     # parent chunk size in tokens
CHUNK_OVERLAP=50           # token overlap between consecutive chunks

# ── Context assembly ──────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS=2000    # token budget for retrieved context
MAX_HISTORY_TOKENS=500     # token budget for conversation history
MAX_RESPONSE_TOKENS=200    # max tokens for LLM to generate

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS=3600     # 1 hour

# ── RRF tuning ────────────────────────────────────────────────────────────────
RRF_K=60                   # standard RRF constant (from original paper)
```

---

## 6. DATABASE SCHEMA (migrations/001_initial_schema.sql)

Run automatically on `docker-compose up` via the `initdb.d` mount.

```sql
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

CREATE INDEX IF NOT EXISTS idx_documents_file_hash  ON documents(file_hash);
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
```

### Why this schema design

- **`access_level` in chunks** — enforced at `WHERE access_level <= $user_clearance` at query time.
  Never post-filter after retrieval (tenant data leak risk if you forget).
- **`parent_chunks` + `chunks`** — two tables because retrieval precision and LLM context quality
  require opposite chunk sizes. Small chunk = precise match. Large parent = rich context.
- **HNSW index** — faster than IVFFlat for < 1M vectors. Set `ef_search=100` at query time for
  better recall. Tune `m` up if recall is too low.
- **`fts_vector` generated column** — eliminates double-write bugs. Always in sync with `content`.

---

## 7. CONFIG (src/config.py)

Use `pydantic-settings`. Every env var maps to a typed field with a sensible default.
The class is a singleton — import `settings` everywhere, never instantiate directly.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    azure_openai_api_version: str = "2024-10-21"

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Cohere
    cohere_api_key: str

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Azure Document Intelligence
    azure_di_endpoint: str = ""
    azure_di_key: str = ""

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Retrieval
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072
    reranker_top_k: int = 20
    final_top_k: int = 5
    chunk_size: int = 512
    parent_chunk_size: int = 2048
    chunk_overlap: int = 50

    # Context assembly
    max_context_tokens: int = 2000
    max_history_tokens: int = 500
    max_response_tokens: int = 200

    # Cache
    cache_ttl_seconds: int = 3600

    # RRF
    rrf_k: int = 60

settings = Settings()
```

---

## 8. DOMAIN MODELS (src/domain/models.py)

All Pydantic v2 models. These are the data contracts used throughout the system.

```
AccessLevel (IntEnum)
    PUBLIC      = 0
    INTERNAL    = 1
    CONFIDENTIAL = 2
    RESTRICTED  = 3

DocumentMeta (BaseModel)
    source_url:   str
    owner_dept:   str
    access_level: AccessLevel = INTERNAL
    tags:         list[str] = []
    metadata:     dict = {}

Document (extends DocumentMeta)
    id:         UUID (auto-generated)
    file_hash:  str           ← SHA-256 of raw file bytes
    created_at: datetime
    updated_at: datetime

ParentChunk (BaseModel)
    id:            UUID
    document_id:   UUID
    content:       str
    page_number:   int | None
    chunk_index:   int
    access_level:  AccessLevel

Chunk (BaseModel)
    id:              UUID
    document_id:     UUID
    parent_chunk_id: UUID | None   ← None when using recursive/semantic strategy
    content:         str
    embedding:       list[float] | None
    page_number:     int | None
    chunk_index:     int
    token_count:     int
    access_level:    AccessLevel
    owner_dept:      str
    source_url:      str
    tags:            list[str]

ScoredChunk (BaseModel)
    chunk:  Chunk
    score:  float
    rank:   int = 0

Citation (BaseModel)
    chunk_id:    UUID
    source_url:  str
    page_number: int | None
    excerpt:     str          ← first 200 chars of the chunk

IngestRequest (BaseModel)          ← used internally, not as HTTP body
    source_url:          str
    owner_dept:          str
    access_level:        AccessLevel = INTERNAL
    tags:                list[str] = []
    chunking_strategy:   str = "recursive"
    use_azure_di:        bool = False

IngestResponse (BaseModel)
    document_id: UUID
    task_id:     str
    status:      str = "queued"

QueryRequest (BaseModel)
    query:                 str
    user_clearance:        AccessLevel = INTERNAL
    owner_dept:            str | None = None    ← filter to specific department
    tags:                  list[str] = []
    use_hyde:              bool = False
    use_multi_query:       bool = False
    conversation_history:  list[dict] = []     ← [{"role": "user", "content": "..."}, ...]

QueryResponse (BaseModel)
    answer:           str
    citations:        list[Citation]
    faithfulness_hint: str | None = None
    latency_ms:       float
    cache_hit:        bool = False
```

### Errors (src/domain/errors.py)

```
RAGError                   ← base
DocumentNotFoundError      ← document ID not in DB
EmbeddingError             ← Azure OpenAI embedding call failed after retries
RerankerError              ← Cohere rerank failed after retries
ContextBudgetExceededError ← all chunks exceed token budget (edge case)
AccessDeniedError          ← user_clearance < document access_level
```

---

## 9. DOCUMENT INGESTION (src/ingestion/)

### 9.1 Loader (loader.py)

Two loaders. Choose at request time via `use_azure_di` flag.

**Loader 1: unstructured.io** (default for most docs)
- Import: `from unstructured.partition.auto import partition`
- Call: `partition(file=io.BytesIO(file_bytes), metadata_filename=filename)`
- Handles: PDF, DOCX, HTML, PPTX, Markdown automatically
- Returns a list of `Element` objects. Each has `.text` and `.metadata.page_number`
- Group elements by page number. Concatenate into one string per page.
- Format output as: `[Page 1]\n<text>\n\n[Page 2]\n<text>...`
- Wrap call with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))`

**Loader 2: Azure Document Intelligence** (for complex layouts, tables, forms)
- Import: `from azure.ai.formrecognizer.aio import DocumentAnalysisClient`
- Model: `"prebuilt-layout"` — extracts text, tables, forms preserving reading order
- Call: `begin_analyze_document("prebuilt-layout", file_bytes)` → await result
- Iterate `result.pages` → for each page, extract `page.lines` → `[line.content for line in page.lines]`
- Format same as above: `[Page N]\n<lines>`
- Wrap with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))`

**RawDocument output object** — both loaders return this:
```python
@dataclass
class RawDocument:
    text:      str        # full document text, page-labeled
    metadata:  dict       # {"pages": N, "filename": "...", "source": "unstructured" or "azure_di"}
    file_hash: str        # SHA-256 of raw file bytes
```

**File hash**: `hashlib.sha256(file_bytes).hexdigest()` — computed before parsing.

### 9.2 Chunker (chunker.py)

**Token counting**: use `tiktoken.get_encoding("cl100k_base")`. Count with `len(enc.encode(text))`.

**Strategy 1: recursive_chunk(text, chunk_size=512, overlap=50)**
- Try separator list in order: `["\n\n", "\n", ". ", " ", ""]`
- For each separator: split text, greedily accumulate until adding next part would exceed chunk_size
- When limit hit: flush current buffer as a chunk, start fresh
- If all chunks fit within chunk_size for this separator, stop and return
- Last separator `""` = hard character split as fallback
- Return: `list[RawChunk]`

**Strategy 2: parent_child_chunk(text, child_size=512, parent_size=2048, overlap=50)**
- First split into PARENT chunks using recursive_chunk with parent_size
- For each parent chunk, split again using recursive_chunk with child_size
- Each child chunk stores `parent_index` pointing to its parent
- Return: `(list[RawChunk] parents, list[RawChunk] children)`
- Children are what gets embedded and stored in `chunks` table
- Parents are stored in `parent_chunks` table
- At query time: retrieve child (precise match) → fetch parent (rich context)

**Strategy 3: semantic_chunk(text, max_chunk_size=512)**
- Split text into sentences using: `re.split(r"(?<=[.!?])\s+", text)`
- Greedily merge sentences: accumulate tokens until next sentence would exceed max_chunk_size
- When over budget: flush current group as chunk, start new group
- Preserves sentence integrity — never splits mid-sentence
- Use for legal/medical documents where breaking a sentence changes meaning

**RawChunk dataclass**:
```python
@dataclass
class RawChunk:
    content:       str
    page_number:   int | None
    chunk_index:   int
    parent_index:  int | None = None  # only set by parent_child strategy
    token_count:   int = 0
```

**chunk_document(text, strategy, chunk_size, parent_chunk_size, overlap)**
- Entry point called by pipeline.py
- Dispatches to correct strategy
- For `"recursive"` and `"semantic"`: return `(chunks, chunks)` — same list for parent and child
- For `"parent_child"`: return `(parent_chunks, child_chunks)` — two separate lists

### 9.3 Pipeline (pipeline.py)

Orchestrates the entire ingestion flow. Constructor takes:
- `pool: asyncpg.Pool`
- `vector_store: VectorStore`
- `bm25_index: BM25Index`

**`pipeline.run(file_bytes, filename, meta, chunking_strategy, use_azure_di)` steps:**

1. Load document via `load_document()` → get `RawDocument`
2. Compute file_hash. Check `vector_store.get_document_by_hash(file_hash)`
   - If found: log "already indexed, skipping" and return existing document_id
3. Create `Document` object. Call `vector_store.upsert_document(doc)`
4. Call `chunk_document()` → get `(parent_raws, child_raws)`
5. Create `ParentChunk` objects. Call `vector_store.upsert_parent_chunks()` → get list of saved parent UUIDs
6. Build texts list from child_raws. Call `get_embeddings(texts)` in batches of 100
7. Build `Chunk` objects — zip child_raws with embeddings. For each child:
   - If `rc.parent_index is not None`: set `parent_chunk_id = parent_ids[rc.parent_index]`
   - Else: `parent_chunk_id = None`
8. Call `vector_store.upsert_chunks(chunks)`
9. Call `bm25_index.add_documents(chunk_ids, texts)` to update keyword index
10. Return `doc.id`

---

## 10. EMBEDDING (src/embedding/encoder.py)

### Model: text-embedding-3-large
- Dimensions: 3072
- Provider: Azure OpenAI
- Client: `AsyncAzureOpenAI` (async, from `openai` package)
- Pass `dimensions=3072` in the request (API supports dimension reduction, always use full)
- Batch size: 100 texts per API call (Azure OpenAI limit)
- Retry: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))`
- Client is created once via `@lru_cache(maxsize=1)` function

### Model: multilingual-e5-large (for GCC Arabic+English deployments)
- Dimensions: 1024
- Library: `sentence-transformers`
- Model ID: `"intfloat/multilingual-e5-large"`
- Important: e5 models require prefix. For passages: `"passage: {text}"`. For queries: `"query: {query}"`
- Call: `model.encode(prefixed_texts, normalize_embeddings=True)` → returns numpy array
- Convert to Python list: `[e.tolist() for e in embeddings]`
- This runs locally (no API call needed)

### Two exported functions:

**`get_embeddings(texts: list[str], model="text-embedding-3-large") -> list[list[float]]`**
- Batch input texts in chunks of 100
- Call appropriate model
- Return flat list of embeddings in same order as input

**`get_query_embedding(query: str, model="text-embedding-3-large") -> list[float]`**
- Single query. Apply `"query: "` prefix for e5 models.
- Returns single embedding (list of floats)

---

## 11. VECTOR STORE (src/store/vector_store.py)

Uses `asyncpg` directly (not SQLAlchemy). Takes `asyncpg.Pool` in constructor.

### Methods to implement:

**`get_document_by_hash(file_hash: str) -> UUID | None`**
- `SELECT id FROM documents WHERE file_hash = $1`
- Returns UUID if found, None otherwise

**`upsert_document(doc: Document) -> None`**
- INSERT with `ON CONFLICT (id) DO UPDATE SET updated_at = now(), file_hash = EXCLUDED.file_hash`
- Pass `doc.metadata` as `json.dumps(doc.metadata)` for the JSONB column

**`upsert_parent_chunks(chunks: list[ParentChunk]) -> list[UUID]`**
- INSERT each chunk and collect the returned `id`
- Return list of UUIDs in same order as input (used by pipeline to set `parent_chunk_id`)

**`upsert_chunks(chunks: list[Chunk]) -> None`**
- INSERT all chunks in a single transaction
- Convert embedding to pgvector format: `f"[{','.join(str(x) for x in embedding)}]"` then cast as `$N::vector`
- Use `ON CONFLICT (id) DO NOTHING`

**`vector_search(query_embedding, user_clearance, owner_dept=None, top_k=20) -> list[ScoredChunk]`**

This is the most important method. The exact SQL:

```sql
SELECT
    id, document_id, parent_chunk_id, content, page_number,
    chunk_index, token_count, access_level, owner_dept, source_url, tags,
    1 - (embedding <=> $1::vector) AS score
FROM chunks
WHERE access_level <= $2
  [AND owner_dept = $4]          -- only added if owner_dept parameter is not None
ORDER BY embedding <=> $1::vector
LIMIT $3
```

Critical: `access_level <= $2` is in the WHERE clause — this is the Row Level Security.
Never filter access after retrieval. The WHERE clause must run before LIMIT.

**`get_parent_content(parent_chunk_id: UUID) -> str | None`**
- `SELECT content FROM parent_chunks WHERE id = $1`
- Called during context assembly to swap child content for richer parent content

---

## 12. BM25 INDEX (src/store/bm25_index.py)

In-memory index. Seeded from DB on startup. Updated incrementally as documents are ingested.

### State
- `_doc_ids: list[str]` — list of chunk UUIDs (as strings), same order as corpus
- `_corpus: list[list[str]]` — tokenized texts (lowercased, whitespace-split)
- `_bm25: BM25Okapi | None` — the rank_bm25 model, rebuilt after every add

### Methods

**`add_documents(doc_ids: list[str], texts: list[str]) -> None`** (async)
- Tokenize: `[text.lower().split() for text in texts]`
- Append to `_doc_ids` and `_corpus`
- Rebuild: `self._bm25 = BM25Okapi(self._corpus)`

**`search(query: str, top_k=20) -> list[tuple[str, float]]`** (sync)
- Tokenize query: `query.lower().split()`
- Call `self._bm25.get_scores(tokens)` → numpy array of scores
- Zip with `_doc_ids`, sort descending by score
- Return top_k as `[(chunk_id_str, score), ...]`
- If `_bm25` is None: return `[]`

**`build_from_db(pool: asyncpg.Pool) -> None`** (async)
- `SELECT id::text, content FROM chunks ORDER BY created_at`
- Call `add_documents()` with all rows
- Called once at application startup in `lifespan` event handler

---

## 13. HYBRID SEARCH WITH RRF (src/retrieval/hybrid.py)

### RRF Function: `reciprocal_rank_fusion(dense_results, bm25_results, k=60) -> list[ScoredChunk]`

RRF formula: `score(document) = Σ 1 / (k + rank_i)` — summed across all result lists where the document appears.

`k=60` is from the original Cormack et al. paper. It dampens the effect of very high-ranked results so results appearing in multiple lists get boosted.

**Implementation steps:**
1. Create `rrf_scores: dict[chunk_id_str, float]` defaulting to 0.0
2. For dense results: for rank `i`, add `1.0 / (k + i + 1)` to `rrf_scores[chunk_id]`
   (rank is 0-indexed, so rank 0 = best)
3. For BM25 results: sort by BM25 score descending, assign ranks 0,1,2,...
   Add `1.0 / (k + rank + 1)` to `rrf_scores[chunk_id]`
4. Sort `rrf_scores` by score descending
5. Map chunk_ids back to `ScoredChunk` objects (use dense_results as the source of Chunk objects)
   — BM25 only has IDs and scores, not full Chunk objects
6. Set `.rank` field from final position

### HybridRetriever class

Constructor: `__init__(self, vector_store: VectorStore, bm25_index: BM25Index)`

**`search(query, query_embedding, user_clearance, owner_dept=None, top_k=20) -> list[ScoredChunk]`**
1. Call `vector_store.vector_search(query_embedding, user_clearance, owner_dept, top_k)` → dense results
2. Call `bm25_index.search(query, top_k)` → BM25 results
3. Call `reciprocal_rank_fusion(dense, bm25, k=settings.rrf_k)`
4. Return `fused[:top_k]`

---

## 14. RERANKER (src/retrieval/reranker.py)

### Why reranking exists

Embeddings and BM25 find candidates based on surface-level similarity. A cross-encoder
(Cohere Rerank) reads query + document together and scores semantic relevance.
Cost: ~350ms added latency. Benefit: faithfulness jumps from 78% → 91%.

### Implementation

**`rerank(query, candidates: list[ScoredChunk], top_n=5) -> list[ScoredChunk]`** (async)

1. Extract texts: `documents = [c.chunk.content for c in candidates]`
2. Call Cohere API:
```python
response = await cohere_client.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=documents,
    top_n=top_n,
    return_documents=False,
)
```
3. `response.results` is a list of `RerankResult` objects, each with:
   - `result.index` — index into original `candidates` list
   - `result.relevance_score` — float 0-1
4. For each result: create `ScoredChunk(chunk=candidates[result.index].chunk, score=result.relevance_score, rank=i)`
5. Fallback: if Cohere fails after retries, log error and `return candidates[:top_n]`

**Cohere client**: `cohere.AsyncClient(api_key=settings.cohere_api_key)` — create once, reuse.

**Retry**: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))`

---

## 15. ADVANCED RETRIEVAL (src/retrieval/advanced.py)

### HyDE (Hypothetical Document Embedding)

**`hyde_embedding(query: str) -> list[float]`** (async)

Problem it solves: when user asks "What is the refund policy?" but documents say
"Customers may request reimbursement within 30 days..." — the query words don't match document words.
HyDE generates a hypothetical answer to the query, then embeds that. The hypothetical answer
uses vocabulary that matches real documents.

Steps:
1. Send to GPT-4o: `"Write a short factual paragraph that directly answers: {query}"`
   `max_tokens=150`, `temperature=0.3`
2. Take the generated text as `hypothetical_answer`
3. Call `get_query_embedding(hypothetical_answer)` — embed the answer, not the query
4. Use this embedding for vector search

### Multi-query Retrieval

**`generate_query_variations(query: str, n=3) -> list[str]`** (async)

Problem it solves: a single query phrasing may miss relevant documents that use different terminology.
Generate N variations → search with each → merge results → more comprehensive recall.

Steps:
1. Prompt GPT-4o: `"Generate {n} different phrasings of this question that preserve the original meaning. Output only questions, one per line, no numbering: {query}"`
2. Split response by newline, strip, take first `n` results
3. Fallback: if empty/failed, return `[query]`

**`merge_results(result_sets: list[list[ScoredChunk]]) -> list[ScoredChunk]`**

Deduplicates by chunk ID, keeps highest score for duplicates:
1. Create dict: `best: dict[chunk_id_str, ScoredChunk]`
2. For each result set, for each ScoredChunk:
   - If chunk_id not in best, or new score > existing score: update best[chunk_id]
3. Sort by score descending, re-assign rank from 0
4. Return merged list

---

## 16. CONTEXT ASSEMBLY (src/context/assembler.py)

**`assemble_context(query, chunks, vector_store, conversation_history=None) -> tuple[str, list[Citation]]`** (async)

Token budget:
- Total budget for context = `settings.max_context_tokens - min(history_token_count, settings.max_history_tokens)`
- e.g. if history uses 200 tokens: context budget = 2000 - 200 = 1800 tokens

For each chunk in reranked order (best first):

1. If chunk has `parent_chunk_id` and parent-child chunking was used:
   - Fetch parent content: `parent_text = await vector_store.get_parent_content(chunk.parent_chunk_id)`
   - Use `parent_text` instead of `chunk.content` for LLM context (richer)
   - Keep using `chunk.content` for citation excerpt
2. Count tokens of content: `count_tokens(content)`
3. If `used_tokens + chunk_tokens > context_budget`:
   - Try LLMLingua compression with remaining budget
   - If compressed version fits: use it. If not: skip this chunk entirely.
4. Format as: `"[Source N: {source_url}, page {page_number}]\n{content}"`
5. Add to `context_parts`. Increment `used_tokens`.
6. Add `Citation(chunk_id, source_url, page_number, excerpt=content[:200])` to citations

Return: `("\n\n---\n\n".join(context_parts), citations)`

### LLMLingua compression

**`_compress_with_llmlingua(text: str, budget: int) -> str | None`**
- Model: `"microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"`
- `target_ratio = max(0.3, budget / count_tokens(text))`
- Call `compressor.compress_prompt(text, rate=target_ratio, force_tokens=["\n"])`
- Return `result["compressed_prompt"]`
- Wrap in try/except: return None on any failure (caller will skip the chunk)

---

## 17. ANSWER GENERATION (src/generation/generator.py)

**`generate_answer(query, context, citations, conversation_history=None) -> str`** (async)

**System prompt** (exact text to use):
```
You are a precise knowledge assistant. Answer questions using ONLY the provided context.
Rules:
- If the answer is not found in the context, respond: "I could not find this information in the available documents."
- Always cite your sources using [Source N] references.
- Be factual, concise, and direct.
- Never fabricate information not present in the context.
```

**Message construction:**
```
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    ...last 4 messages from conversation_history (2 turns),
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (cite sources as [Source N]):"}
]
```

**API call:**
```python
response = await client.chat.completions.create(
    model=settings.azure_openai_deployment,
    messages=messages,
    max_tokens=settings.max_response_tokens,
    temperature=0.1,   # Low temperature for factual grounded answers
)
```

Return `response.choices[0].message.content`
Fallback: if content is None or empty, return `"I could not find this information in the available documents."`

Retry with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))`

---

## 18. REDIS CACHE (src/cache/query_cache.py)

Cache key formula:
```python
key_data = f"{query}|{user_clearance}|{owner_dept or ''}"
cache_key = "rag:query:" + hashlib.sha256(key_data.encode()).hexdigest()
```

**`get(query, user_clearance, owner_dept) -> dict | None`**
- `await redis.get(cache_key)` → decode JSON if exists

**`set(query, user_clearance, owner_dept, result: dict) -> None`**
- `await redis.setex(cache_key, settings.cache_ttl_seconds, json.dumps(result))`
- Store the full `QueryResponse.model_dump(mode="json")` result

**`invalidate_pattern(pattern="rag:query:*") -> int`**
- Delete all keys matching pattern (called after re-index to flush stale cached answers)

### Why cache by (query + clearance + dept)
Two users with different clearance levels asking the same question get different results
(because access-controlled retrieval returns different chunks). The cache key must include
clearance to avoid leaking restricted content to lower-clearance users.

---

## 19. BACKGROUND INDEXING (src/indexing/)

### worker.py — Celery app factory

```python
celery_app = Celery(
    "rag_indexing",
    broker=settings.celery_broker_url,      # redis://localhost:6379/1
    backend=settings.celery_result_backend, # redis://localhost:6379/2
    include=["src.indexing.tasks"],
)

# Task routing
task_routes = {
    "src.indexing.tasks.ingest_document_task": {"queue": "ingest"},
    "src.indexing.tasks.full_reindex_task":    {"queue": "reindex"},
}

# Periodic schedules
beat_schedule = {
    "full-reindex-weekly": {
        "task": "src.indexing.tasks.full_reindex_task",
        "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2am UTC
    },
    "incremental-sync-daily": {
        "task": "src.indexing.tasks.incremental_sync_task",
        "schedule": crontab(hour=3, minute=0),                  # Daily 3am UTC
    },
}
```

### tasks.py — Celery tasks

**`ingest_document_task(file_path, source_url, owner_dept, access_level=1, tags=[], chunking_strategy="recursive", use_azure_di=False)`**

This is a sync Celery task that wraps async pipeline code:
1. Read file bytes from `file_path`
2. Build `DocumentMeta` from parameters
3. Create a new asyncio event loop: `loop = asyncio.new_event_loop()`
4. Run the async ingestion pipeline: `loop.run_until_complete(pipeline.run(...))`
5. Log result, return `str(doc_id)`
6. On exception: `raise self.retry(exc=exc, countdown=60)` (max_retries=3)

**`incremental_sync_task(watch_dir="/app/docs_watch")`**

File change detection:
1. Get all known file hashes from DB: `SELECT source_url, file_hash FROM documents`
2. Walk `watch_dir` with `Path(watch_dir).rglob("*")`, skipping directories
3. For each file: compute `hashlib.sha256(file.read_bytes()).hexdigest()`
4. If `source_url` not in known_hashes OR hash is different → queue `ingest_document_task.delay(...)`
5. This detects both new files and modified files (hash changes on modification)

**`full_reindex_task()`**
- Runs weekly on Sunday 2am UTC (off-peak)
- Drops all chunk embeddings, re-embeds from stored content
- In practice: fetch all `(id, content)` from chunks table, re-compute embeddings, UPDATE

---

## 20. TRACING (src/tracing/tracer.py)

### RAGTrace class

Created at the start of each query request. Wraps Langfuse tracing.

```python
class RAGTrace:
    def __init__(self, query: str, user_clearance: int):
        # Create Langfuse trace: lf.trace(name="rag-query", input={"query": query, "clearance": user_clearance})
        # Store trace object and empty spans dict

    def span(self, name: str, input_data: dict | None = None):
        # Create child span: self._trace.span(name=name, input=input_data)
        # Store in self._spans[name]
        # If Langfuse not configured: return NullSpan()

    def end_span(self, name: str, output: dict | None = None, metadata: dict | None = None):
        # End the named span: span.end(output=output, metadata=metadata)

    def end(self, output: dict | None = None):
        # End the root trace
```

**Span names used in query route** (in order):
1. `"embedding"` — input: `{query, use_hyde}` | output: nothing (embedding is opaque)
2. `"retrieval"` — input: `{use_multi_query}` | output: `{candidates: N}`
3. `"rerank"` — input: `{candidates: N}` | output: `{reranked: N}`
4. `"assemble"` — no input | output: `{tokens_used: N}`
5. `"generate"` — no input | output: `{answer_length: N}`

**_NullSpan**: A no-op object with `.end(**kwargs)` that does nothing. Returned when Langfuse is not configured.

**Langfuse client init**: lazy, on first use. If `settings.langfuse_public_key` is empty string → skip. Wrap in try/except and log warning if import fails.

---

## 21. API LAYER (src/api/)

### main.py — FastAPI app factory

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup:
    #   1. await get_db_pool()        — warm the asyncpg connection pool
    #   2. await get_bm25_index()     — seed BM25 from DB
    yield
    # Shutdown: nothing to do (pool closes automatically)

app = FastAPI(title="Enterprise RAG Knowledge Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
```

### deps.py — Dependency injection

Module-level singletons created lazily (created once, reused):
- `_pool: asyncpg.Pool | None` — created by `get_db_pool()`
- `_bm25: BM25Index | None` — created by `get_bm25_index()`

**`get_db_pool() -> asyncpg.Pool`** (async)
- If `_pool is None`: `await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)`
- Return `_pool`

**`get_vector_store() -> VectorStore`** (async)
- `pool = await get_db_pool()`
- Return `VectorStore(pool)`

**`get_bm25_index() -> BM25Index`** (async)
- If `_bm25 is None`: create `BM25Index()`, call `await bm25.build_from_db(pool)`
- Return `_bm25`

**`get_hybrid_retriever() -> HybridRetriever`** (async)
- Get vector store + BM25 index, return `HybridRetriever(vs, bm25)`

**`get_redis() -> aioredis.Redis`** (async)
- `return aioredis.from_url(settings.redis_url, decode_responses=True)`

**`get_query_cache() -> QueryCache`** (async)
- Get redis client, return `QueryCache(redis)`

**`get_ingestion_pipeline() -> IngestionPipeline`** (async)
- Get pool, VectorStore, BM25Index, return `IngestionPipeline(pool, vs, bm25)`

### routes/health.py

```
GET /health        → 200 {"status": "ok"}
GET /health/ready  → 200 {"status": "ready", "db": "ok", "cache": "ok"}
                     503 {"detail": "<error message>"} if DB or Redis unreachable
```

Readiness check: `SELECT 1` against DB + `redis.ping()`.

### routes/ingest.py

```
POST /v1/ingest
Content-Type: multipart/form-data

Fields:
  file           (UploadFile, required)
  source_url     (str, required)
  owner_dept     (str, required)
  access_level   (int, default=1)
  tags           (str, default="", comma-separated)
  chunking_strategy (str, default="recursive")
  use_azure_di   (bool, default=false)

Response 200:
  {
    "document_id": "uuid",
    "task_id":     "uuid",
    "status":      "queued"
  }
```

Implementation: read file bytes, build DocumentMeta, add `pipeline.run(...)` as a FastAPI `BackgroundTask`.
Return immediately with status "queued" — ingestion runs in background.

### routes/query.py

```
POST /v1/query
Content-Type: application/json

Request:
  {
    "query": "What is the refund policy for enterprise customers?",
    "user_clearance": 1,
    "owner_dept": "finance",          (optional — filter to one department)
    "tags": [],
    "use_hyde": false,
    "use_multi_query": false,
    "conversation_history": [
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }

Response 200:
  {
    "answer": "Enterprise customers may request refunds within 30 days [Source 1].",
    "citations": [
      {
        "chunk_id": "uuid",
        "source_url": "https://internal.company.com/docs/refund-policy.pdf",
        "page_number": 3,
        "excerpt": "Enterprise customers may request reimbursement..."
      }
    ],
    "latency_ms": 1240.5,
    "cache_hit": false
  }
```

**Full query route implementation — step by step:**
1. Record `start = time.perf_counter()`
2. Create `trace = RAGTrace(request.query, request.user_clearance.value)`
3. Check cache: `cached = await cache.get(...)` — if hit, return immediately with `cache_hit=True`
4. Get query embedding (HyDE branch or standard)
5. Get candidates (multi-query branch or standard single search)
6. If `candidates` is empty: return "could not find relevant information" response
7. Rerank: `reranked = await rerank(query, candidates, top_n=settings.final_top_k)`
8. Assemble context: `context, citations = await assemble_context(...)`
9. Generate: `answer = await generate_answer(...)`
10. Compute `latency_ms = (time.perf_counter() - start) * 1000`
11. Build `QueryResponse`, cache it, end trace, return

Wrap spans around steps 4–9 using `trace.span(name)` / `trace.end_span(name)`.

---

## 22. EVALUATION (eval/)

### golden_dataset.json format

200 Q&A pairs. Each case:
```json
{
  "question": "What is the maximum claim amount under the enterprise support policy?",
  "ground_truth": "Enterprise support customers can claim up to $50,000 per incident...",
  "relevant_document": "support-policy-v3.pdf",
  "difficulty": "medium"
}
```

For the initial build, include at least 10 cases covering:
- Direct factual lookups
- Multi-hop (answer requires combining 2 sources)
- Negative cases (answer not in knowledge base — should say "not found")
- Access control cases (restricted documents)

### ragas_config.py

```python
FAITHFULNESS_THRESHOLD    = 0.85   # fraction of claims in answer grounded in context
ANSWER_RELEVANCE_THRESHOLD = 0.80  # answer addresses the question
CONTEXT_PRECISION_THRESHOLD = 0.75 # relevant chunks are ranked before irrelevant ones

METRICS_TO_RUN = ["faithfulness", "answer_relevancy", "context_precision"]
```

### run_eval.py

Loads golden dataset → for each question runs the full query pipeline → collects
`(question, answer, contexts, ground_truth)` → evaluates with RAGAS → prints results table.

Exit codes: `sys.exit(1)` if faithfulness < 0.85 or answer_relevancy < 0.80. `sys.exit(0)` otherwise.

The CI pipeline calls `python eval/run_eval.py` and the non-zero exit code blocks the merge.

---

## 23. DOCKER SETUP

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured.io (PDF parsing, OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY migrations/ migrations/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

5 services:

**api** — the FastAPI app
- Build from Dockerfile
- Port: 8000:8000
- env_file: .env
- Depends on postgres (healthy) + redis (healthy)
- Volume: `./docs_watch:/app/docs_watch` for incremental sync watcher

**worker** — Celery worker
- Same image as api
- Command: `celery -A src.indexing.worker worker --loglevel=info -Q ingest,reindex`
- env_file: .env
- Depends on postgres + redis

**beat** — Celery beat scheduler (periodic tasks)
- Same image as api
- Command: `celery -A src.indexing.worker beat --loglevel=info`
- env_file: .env
- Depends on redis (NOT postgres — beat only pushes to queue)

**postgres** — pgvector-enabled PostgreSQL
- Image: `pgvector/pgvector:pg16` (not plain postgres — must have this image for vector extension)
- Env: `POSTGRES_USER=rag`, `POSTGRES_PASSWORD=secret`, `POSTGRES_DB=ragdb`
- Port: 5432:5432
- Volume: `pgdata:/var/lib/postgresql/data`
- Volume: `./migrations:/docker-entrypoint-initdb.d` (auto-runs SQL files on first start)
- Healthcheck: `pg_isready -U rag -d ragdb`

**redis** — cache + message broker
- Image: `redis:7-alpine`
- Port: 6379:6379
- Healthcheck: `redis-cli ping`

---

## 24. CI/CD PIPELINES (.github/workflows/)

### ci.yml — runs on every push to any branch

```yaml
on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: pytest tests/unit -v
```

### ragas-eval-gate.yml — runs on PR to main only

```yaml
on:
  pull_request:
    branches: [main]

jobs:
  ragas-eval:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: rag
          POSTGRES_PASSWORD: secret
          POSTGRES_DB: ragdb
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U rag -d ragdb"
          --health-interval 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: psql postgresql://rag:secret@localhost:5432/ragdb -f migrations/001_initial_schema.sql
      - run: python eval/run_eval.py
        env:
          AZURE_OPENAI_ENDPOINT:  ${{ secrets.AZURE_OPENAI_ENDPOINT }}
          AZURE_OPENAI_API_KEY:   ${{ secrets.AZURE_OPENAI_API_KEY }}
          COHERE_API_KEY:         ${{ secrets.COHERE_API_KEY }}
          LANGFUSE_PUBLIC_KEY:    ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY:    ${{ secrets.LANGFUSE_SECRET_KEY }}
          DATABASE_URL:           postgresql://rag:secret@localhost:5432/ragdb
          REDIS_URL:              redis://localhost:6379/0
          CELERY_BROKER_URL:      redis://localhost:6379/1
          CELERY_RESULT_BACKEND:  redis://localhost:6379/2
```

---

## 25. TESTS

### conftest.py

Fixtures needed:
- `pg_pool` — asyncpg pool connected to test DB (use a separate `ragdb_test` database)
- `vector_store` — `VectorStore(pg_pool)`
- `bm25_index` — fresh empty `BM25Index()`
- `mock_embeddings` — returns deterministic `[0.1] * 3072` (avoids real API calls in unit tests)
- `sample_chunk` — a `Chunk` object with fixed UUID for assertion

### tests/unit/test_chunker.py

Test cases to write:
1. `test_recursive_chunk_respects_token_size` — all chunks must have token_count <= chunk_size + 20% tolerance
2. `test_recursive_chunk_overlap` — consecutive chunks share some tokens (verify text overlap exists)
3. `test_parent_child_children_reference_correct_parent` — each child.parent_index points to valid parent
4. `test_parent_child_parent_covers_all_children` — concatenated children text should appear in parent text
5. `test_semantic_chunk_no_mid_sentence_splits` — each chunk ends with `.`, `!`, or `?`
6. `test_empty_text_returns_empty` — all strategies return empty list for empty string
7. `test_chunk_document_dispatch` — correct strategy called based on strategy parameter

### tests/unit/test_hybrid.py

1. `test_rrf_boosts_document_in_both_lists` — doc appearing in dense+BM25 has higher score than doc in only one
2. `test_rrf_k60_dampening` — rank-1 vs rank-10 score difference follows RRF formula
3. `test_rrf_returns_only_chunks_from_dense_results` — BM25-only docs (not in dense) are excluded from output
4. `test_rrf_empty_bm25` — handles empty BM25 results gracefully (returns dense ranking only)
5. `test_rrf_empty_dense` — handles empty dense results gracefully (returns empty)

### tests/unit/test_reranker.py

1. `test_reranker_fallback_on_error` — mock Cohere to raise exception → verify returns `candidates[:top_n]`
2. `test_reranker_respects_top_n` — output length <= top_n
3. `test_reranker_preserves_chunk_objects` — reranked chunks are same objects as input (by chunk.id)

### tests/integration/test_ingest.py

Run against real test DB (pgvector must be available):
1. `test_full_ingest_stores_chunks` — ingest sample PDF bytes → verify chunks in DB
2. `test_dedup_on_same_file_hash` — ingest same file twice → only one document row
3. `test_access_level_stored_correctly` — ingest with RESTRICTED level → verify in DB

### tests/integration/test_query.py

1. `test_query_returns_answer` — ingest a doc, query it, verify answer is not empty
2. `test_access_control_filters_results` — ingest RESTRICTED doc, query with INTERNAL clearance → 0 results
3. `test_cache_hit_returns_same_answer` — query twice with same params → second response has `cache_hit=True`

---

## 26. MONITORING & ALERTING

These are infrastructure configs, not code. Document them so the AI knows what to set up.

### Grafana dashboards (connect to Langfuse + Prometheus metrics from FastAPI)

Panels to create:
- `retrieval_precision_at_k` — % of top-K results that were relevant (from RAGAS)
- `faithfulness_score` — rolling 1h average faithfulness (from RAGAS eval runs)
- `p95_latency_ms` — P95 query latency broken down by: with reranker / cache hit / cache miss
- `cost_per_query` — Azure OpenAI token cost + Cohere API cost per query
- `cache_hit_rate` — Redis hit rate over time

### PagerDuty alert rule

Alert condition: `faithfulness_score < 0.85` sustained for 5 minutes → page on-call.

### Langfuse (automatic — just configure keys)

Langfuse records every span automatically. View full traces at `cloud.langfuse.com`.
Each query shows: embedding latency | retrieval count | rerank score | context tokens | generation latency.

---

## 27. KEY DESIGN DECISIONS (know these for interviews)

### Decision 1: Access control in WHERE clause, never post-filter

BAD approach: retrieve top-20 chunks for all users, then filter to user's clearance level.
Problem: if filtering removes most results, you end up with fewer than top-K. Also: a bug
in post-filtering could leak restricted documents to lower-clearance users.

CORRECT approach: `WHERE access_level <= $user_clearance` in the SQL query itself.
The vector index search only ever looks at documents the user is allowed to see.

### Decision 2: Parent-child chunking for technical documents

Small chunks (512 tokens) get better embedding precision — the embedding is focused.
Large chunks (2048 tokens) give the LLM more context — it can see the full explanation.
The trick: embed the small chunk for retrieval accuracy, but pass the parent chunk to the LLM.

### Decision 3: Cache by (query + clearance + dept), not just query

Same query asked by a level-1 and level-3 user returns different documents (different access).
If you cache by query string only, level-1 user could get level-3 cached results.

### Decision 4: Reranker is optional path, cached for top-500 queries

Reranker adds 350ms. For frequently-asked queries, Redis cache eliminates this cost.
Cache covers 38% of production traffic → 38% of queries pay 0ms reranker latency.

### Decision 5: Incremental indexing via file hash, not file modification time

Modification timestamps can be unreliable (file copies, backups, timezone issues).
SHA-256 hash is always deterministic — same bytes = same hash = skip re-indexing.

### Decision 6: BM25 index in-memory, seeded from DB on startup

rank_bm25 is fast but in-memory. On restart: rebuild from DB `SELECT id, content FROM chunks`.
For very large indexes (>5M chunks): switch to Elasticsearch, which persists its own index.

---

## 28. FAILURE MODES AND FIXES

| Failure | Symptom | Fix |
|---------|---------|-----|
| Stale docs in index | Answers reference deleted/updated content | Incremental indexing via file hash change detection on `docs_watch/` directory |
| Tenant data leak | User sees another department's docs | Access control in WHERE clause at query time, never post-retrieval filter |
| Hallucination despite retrieval | Answer adds facts not in retrieved chunks | System prompt: "only answer from context, say 'not found' if not in context" |
| Chunk size too small | Low context quality, LLM gets fragmented info | Use parent-child chunking: small child for retrieval, parent for LLM context |
| Chunk size too large | Low retrieval precision, irrelevant chunks retrieved | Reduce chunk_size. Evaluate with context_precision@k metric |
| Reranker latency too high | P95 > 3s | Cache reranker results. 38% of traffic hits cache = 350ms saved |
| BM25 out of sync after crash | Keyword search misses recently-added docs | `build_from_db()` on every startup rebuilds BM25 from source of truth |
| Low faithfulness score (<0.85) | RAGAS alert fires, CI gate fails | Check top-5 reranked chunks for a failing query in Langfuse. Usually: wrong chunks retrieved → tune embedding or reranker. Or: chunk too long → use LLMLingua |
| High latency on cold start | First query slow (BM25 rebuild) | Pre-warm during `lifespan` startup. Async pool and BM25 both initialized before serving traffic |

---

## 29. BUILD ORDER FOR AN AI CODING TOOL

Build in this exact order to avoid import failures:

```
Step 1:  pyproject.toml + .env.example + .gitignore
Step 2:  migrations/001_initial_schema.sql
Step 3:  src/config.py
Step 4:  src/domain/models.py + src/domain/errors.py
Step 5:  src/embedding/encoder.py
Step 6:  src/store/vector_store.py + src/store/bm25_index.py
Step 7:  src/ingestion/loader.py + src/ingestion/chunker.py + src/ingestion/pipeline.py
Step 8:  src/retrieval/hybrid.py + src/retrieval/reranker.py + src/retrieval/advanced.py
Step 9:  src/context/assembler.py
Step 10: src/generation/generator.py
Step 11: src/cache/query_cache.py
Step 12: src/tracing/tracer.py
Step 13: src/indexing/worker.py + src/indexing/tasks.py
Step 14: src/api/deps.py
Step 15: src/api/routes/health.py + ingest.py + query.py
Step 16: src/api/main.py
Step 17: eval/ragas_config.py + eval/golden_dataset.json + eval/run_eval.py
Step 18: tests/conftest.py + all test files
Step 19: Dockerfile + docker-compose.yml
Step 20: .github/workflows/ci.yml + ragas-eval-gate.yml
```

---

## 30. QUICK START (how to run locally after build)

```bash
# 1. Copy env file and fill in your keys
cp .env.example .env

# 2. Start all services
docker-compose up -d

# 3. Wait for postgres to be ready, then verify schema was applied
docker-compose exec postgres psql -U rag -d ragdb -c "\dt"

# 4. Test health
curl http://localhost:8000/health/ready

# 5. Ingest a document
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@sample.pdf" \
  -F "source_url=https://internal.company.com/docs/sample.pdf" \
  -F "owner_dept=engineering" \
  -F "access_level=1"

# 6. Query the knowledge base
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "user_clearance": 1}'
```
