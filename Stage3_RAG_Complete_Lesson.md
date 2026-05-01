# Stage 3: Retrieval-Augmented Generation — Complete Senior-Level Lesson

> **Three layers per topic:**
> **Concept Layer** — how it actually works, plain English first, then deep mechanics
> **Engineering Layer** — production code, edge cases, what breaks
> **Architecture Layer** — system design, patterns, scaling, tradeoffs

---

**Central concept of this stage:** RAG = give the model the right documents at the right time — retrieval quality determines answer quality more than model quality.

Stage 2 taught you how to call an LLM: prompt design, structured outputs, tool calling, evaluation. Stage 3 teaches you how to give the model *knowledge it was never trained on* — without retraining it. Every topic builds one piece of the retrieval pipeline: deciding when RAG is the right tool, embedding documents, searching them at scale, assembling context under a token budget, and measuring whether the system actually answers correctly. The output of this stage is an enterprise-grade knowledge assistant that cites its sources, enforces per-tenant access control, and scores itself with the RAGAS evaluation framework.

### Topic Dependency Map

```
RAG: Give the model the right documents at the right time
      │
      ├── 3.1 Why RAG Exists ──────────────────── foundational judgment
      │
      ├── 3.2 Embeddings ──→ 3.3 Model Selection
      │        └── semantic representation foundation
      │                         ↓
      ├── 3.4 Vector Search ──→ 3.5 Vector DB Selection
      │        └── how search works    └── where to store
      │
      ├── 3.6 Ingestion ──→ 3.7 Chunking ──→ 3.8 Metadata Design
      │        └── get docs in         └── split right    └── filter later
      │
      ├── 3.9 Retrieval Patterns ──→ 3.10 Prompt Assembly ──→ 3.11 Citations
      │        └── find chunks             └── pack context      └── show sources
      │
      ├── 3.12 Access Control ──→ 3.13 Index Freshness
      │        └── who sees what       └── keep it current
      │
      ├── 3.14 RAG Evaluation ──→ 3.15 Failure Modes
      │        └── measure quality         └── fix what breaks
      │
      ├── 3.16 Advanced RAG ──→ 3.17 Hybrid Architectures
      │        └── improve retrieval      └── multi-source knowledge
      │
      └── 3.18 Enterprise Project ── integrates everything above
```

---

## Table of Contents
- [3.1 Why RAG Exists](#31-why-rag-exists)
- [3.2 Embeddings](#32-embeddings)
- [3.3 Embedding Model Selection](#33-embedding-model-selection)
- [3.4 Vector Search Basics](#34-vector-search-basics)
- [3.5 Vector Database Selection](#35-vector-database-selection)
- [3.6 Document Ingestion Pipelines](#36-document-ingestion-pipelines)
- [3.7 Chunking Strategies](#37-chunking-strategies)
- [3.8 Metadata Design](#38-metadata-design)
- [3.9 Retrieval Patterns](#39-retrieval-patterns)
- [3.10 Prompt Assembly and Context Packing](#310-prompt-assembly-and-context-packing)
- [3.11 Citation and Grounding Patterns](#311-citation-and-grounding-patterns)
- [3.12 Access Control and Tenant Isolation](#312-access-control-and-tenant-isolation)
- [3.13 Index Freshness](#313-index-freshness)
- [3.14 RAG Evaluation](#314-rag-evaluation)
- [3.15 RAG Failure Modes](#315-rag-failure-modes)
- [3.16 Advanced RAG Patterns](#316-advanced-rag-patterns)
- [3.17 Hybrid Architectures](#317-hybrid-architectures)
- [3.18 Stage 3 Project: Enterprise Knowledge Assistant](#318-stage-3-project-enterprise-knowledge-assistant)
- [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## 3.1 Why RAG Exists

### 🧠 Mental Model
> A fine-tuned model learns new *behavior*. Prompt stuffing adds *context inline*. RAG retrieves *the right documents* at query time. The judgment question is: do you need new skills, new facts, or dynamic facts? Each answer points to a different tool.

**Connects to:** Stage 2 hallucinations → **Why RAG Exists** → Embeddings (3.2)
**Parent concept:** Knowledge injection strategies
**Builds on:** Section 2.10 — hallucinations occur when the model lacks grounding; RAG is the production answer to that problem

---

### Concept Layer

#### The Problem RAG Solves

Your model was trained on data with a cutoff date. It knows nothing about your internal documents, your product catalog, your customer tickets, or anything that happened last week. When you ask it about these things, it either hallucinates or refuses.

You have three tools to fix this. Each one solves a different version of the problem.

**Tool 1: Prompt stuffing.** Paste the documents directly into the prompt. Simple, works immediately, but limited by context window size. A 128K context window holds roughly 90,000 words — one long document, not a knowledge base of 50,000 documents.

**Tool 2: Fine-tuning.** Retrain the model on your documents. The knowledge becomes baked into weights. But fine-tuning teaches *patterns*, not facts. A model fine-tuned on your support documentation learns to write in your support tone — it does not reliably memorize specific policy numbers or version strings. Fine-tuned models also cannot update when your documents change without retraining.

**Tool 3: RAG.** At query time, retrieve the relevant documents and inject only those into the prompt. The model sees exactly the context it needs. Documents update independently of the model.

**The rule:** Use prompt stuffing when the context fits and changes rarely. Use fine-tuning when you need new skills or tone, not new facts. Use RAG when your knowledge base is large, changes frequently, or must be cited.

#### The Decision Framework

Here is the exact judgment question to ask at each step:

```
Step 1: Does the model lack knowledge it needs?
   No → It is a prompt design problem, not a knowledge problem (go back to Stage 2)
   Yes → continue

Step 2: How big is the knowledge base?
   < 20 pages and static → Prompt stuffing. Put it in the system prompt. Done.
   > 20 pages or dynamic → continue

Step 3: Does the knowledge change more than monthly?
   No, and it fits in fine-tune training data → Consider fine-tuning for tone/skill
   Yes → RAG is the answer

Step 4: Do you need to cite sources?
   Yes → RAG (you can trace which chunk produced which answer)
   No → either works
```

**In short:** Prompt stuffing for small static context, fine-tuning for new skills, RAG for large or dynamic knowledge bases that need citations.

#### Why Fine-Tuning Fails at Fact Injection

This is the most common misconception in LLM engineering. Engineers assume fine-tuning is "teaching the model facts." It is not.

Fine-tuning adjusts weights across the entire network. When you fine-tune on a document that says "our refund window is 30 days," the model does not store "30 days" in a retrievable slot. It learns that "refund window" questions should be answered in a certain register and style. The actual number gets distributed across billions of parameters in a way that makes it statistically likely but not guaranteed. Ask the fine-tuned model 1,000 times — it might say "30 days" 900 times and "60 days" 100 times.

RAG retrieves the exact sentence. The model then reads it and reports it. That is why RAG has better factual accuracy for specific facts.

**The rule:** Fine-tune to change how the model responds. Use RAG to change what the model knows.

#### When Fine-Tuning + RAG Works Together

These are not mutually exclusive. The combination that works best:

1. Fine-tune for tone, structure, and domain vocabulary
2. Use RAG to inject the actual facts

Example: A legal AI assistant. Fine-tune on legal writing style so the model structures responses like a lawyer. Use RAG to inject the actual case law and contract clauses. Neither alone gives you both.

**In short:** Fine-tuning and RAG solve different problems. Use fine-tuning to change behavior; use RAG to change knowledge. Use both when you need both.

---

### Engineering Layer

#### The Decision Function in Code

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class KnowledgeStrategy(Enum):
    PROMPT_STUFFING = "prompt_stuffing"
    FINE_TUNING = "fine_tuning"
    RAG = "rag"
    FINE_TUNING_PLUS_RAG = "fine_tuning_plus_rag"


@dataclass
class KnowledgeRequirements:
    document_count: int           # total documents in knowledge base
    avg_document_tokens: int      # average tokens per document
    update_frequency_days: int    # how often docs change (0 = never)
    needs_citation: bool          # must source be traceable?
    needs_tone_shift: bool        # must model adopt domain style?
    context_window_tokens: int    # model's max context (e.g. 128000)


def select_knowledge_strategy(req: KnowledgeRequirements) -> tuple[KnowledgeStrategy, str]:
    total_tokens = req.document_count * req.avg_document_tokens
    fits_in_context = total_tokens < (req.context_window_tokens * 0.6)  # 60% headroom

    if fits_in_context and req.update_frequency_days == 0:
        return (
            KnowledgeStrategy.PROMPT_STUFFING,
            f"Total {total_tokens} tokens fits in context window. Static content. No infrastructure needed."
        )

    if req.update_frequency_days > 0 or req.needs_citation:
        if req.needs_tone_shift:
            return (
                KnowledgeStrategy.FINE_TUNING_PLUS_RAG,
                "Dynamic docs + citation + tone shift: fine-tune for style, RAG for facts."
            )
        return (
            KnowledgeStrategy.RAG,
            f"{req.document_count} docs, updates every {req.update_frequency_days}d, "
            f"citation={'required' if req.needs_citation else 'not required'}. Use RAG."
        )

    if req.needs_tone_shift and not req.needs_citation:
        return (
            KnowledgeStrategy.FINE_TUNING,
            "Static knowledge + tone shift needed. Fine-tune. Re-run when docs change."
        )

    return (
        KnowledgeStrategy.RAG,
        "Default to RAG — it handles change and scale without retraining."
    )


# Example usage
if __name__ == "__main__":
    req = KnowledgeRequirements(
        document_count=5000,
        avg_document_tokens=800,
        update_frequency_days=7,
        needs_citation=True,
        needs_tone_shift=False,
        context_window_tokens=128_000,
    )
    strategy, reason = select_knowledge_strategy(req)
    print(f"Strategy: {strategy.value}")
    print(f"Reason: {reason}")
    # Strategy: rag
    # Reason: 5000 docs, updates every 7d, citation=required. Use RAG.
```

The function encodes the decision tree explicitly so every engineer on the team applies the same logic. The 60% headroom constant prevents prompt stuffing from crowding out the model's own reasoning space — a common mistake that causes response quality to degrade near the context limit.

---

### Architecture Layer

#### The Three Strategies Side by Side

```
PROMPT STUFFING
──────────────────────────────────────────────────────────
User Query ──→ [All Documents in Prompt] ──→ LLM ──→ Answer
                     ↑
              (static, fits in context window)
              Cost: O(n × tokens) per query
              Update lag: 0 (change the system prompt)

FINE-TUNING
──────────────────────────────────────────────────────────
Documents ──→ [Training Pipeline] ──→ Fine-tuned Model Weights
                                              ↓
User Query ────────────────────────────→ LLM ──→ Answer
              Cost: one-time training ($50–$500 for 7B models)
              Update lag: full retrain cycle (hours to days)

RAG
──────────────────────────────────────────────────────────
Documents ──→ [Embed + Index] ──→ Vector DB
                                      ↓
User Query ──→ [Embed Query] ──→ [Retrieve Top-k] ──→ LLM ──→ Answer + Citations
              Cost: embedding per query + LLM call
              Update lag: document re-embedding time (seconds to minutes)
```

| Dimension | Prompt Stuffing | Fine-Tuning | RAG |
|-----------|----------------|-------------|-----|
| Setup time | Minutes | Days | Hours |
| Knowledge update | Edit system prompt | Retrain | Re-embed doc |
| Max knowledge size | ~60K tokens | Unlimited | Unlimited |
| Citation support | Manual | No | Yes |
| Factual accuracy | High (verbatim) | Medium | High |
| Infrastructure cost | None | GPU training | Vector DB |
| Latency added | +0ms | +0ms | +50–200ms |

> **Bridge:** You have decided to use RAG. The first thing you need is a way to represent the *meaning* of documents as numbers — so you can search by meaning, not by keyword. That is what embeddings do.

---

⚡ **Senior Checklist — 3.1**
- [ ] Never use fine-tuning to inject facts — test with 100 queries and measure fact recall before committing
- [ ] Calculate total tokens before choosing prompt stuffing: `doc_count × avg_tokens × 0.6 < context_window`
- [ ] Budget update latency: fine-tuning pipelines take 4–48 hours; RAG re-indexing takes seconds per document
- [ ] For citation requirements, RAG is the only viable path — fine-tuned models cannot trace which training example produced which output
- [ ] Combine fine-tuning + RAG when you need both domain tone and current facts — they solve different layers
- [ ] Default to RAG over prompt stuffing above 20 documents: even if it fits today, the knowledge base will grow
- [ ] Document the strategy decision and its reasoning in your architecture notes — future engineers will ask why

---

## 3.2 Embeddings

### 🧠 Mental Model
> An embedding is a point in high-dimensional space. Two texts that mean the same thing land near each other. Two texts that mean different things land far apart. The distance between two points is the semantic distance between two ideas.

**Connects to:** Why RAG Exists (3.1) → **Embeddings** → Embedding Model Selection (3.3)
**Parent concept:** Semantic representation
**Builds on:** Section 3.1 — RAG requires searching by meaning; embeddings are the mechanism that makes meaning searchable

---

### Concept Layer

#### From Words to Coordinates

A keyword search engine finds documents containing the exact word "refund." It misses documents that say "money back" or "return policy" — same meaning, different words. Embeddings solve this.

An embedding model converts text into a list of numbers — a vector. For `text-embedding-3-small`, that list has 1,536 numbers. For `BGE-M3`, it has 1,024. The specific numbers don't matter; what matters is their *relationships*. The model is trained so that semantically similar texts produce numerically similar vectors.

```
"How do I cancel my subscription?"
    → [0.021, -0.134, 0.887, ..., 0.042]  # 1,536 numbers

"Steps to end my membership"
    → [0.019, -0.131, 0.891, ..., 0.039]  # very similar numbers

"How to bake sourdough bread"
    → [-0.432, 0.762, -0.201, ..., 0.817]  # very different numbers
```

The numbers for the first two sentences are close together because they mean the same thing. The bread sentence is far away because it means something completely different.

**In short:** Embeddings turn text into coordinates. Similar meaning = nearby coordinates. Different meaning = distant coordinates.

#### How Embedding Models Learn

Embedding models are trained with **contrastive learning**. The training process shows the model pairs of sentences:

- "My order never arrived" and "Package not delivered" → these should be **close** (positive pair)
- "My order never arrived" and "I love this product" → these should be **far** (negative pair)

The model learns to produce vectors that satisfy these distance constraints across millions of pairs. After training, it generalizes: it has never seen your specific documents, but it produces meaningful coordinates for them because the semantic structure of language is consistent.

**The rule:** Embedding quality is entirely determined by training data. A model trained on English forum posts will not embed technical German contracts well. Choose your embedding model based on your domain and language.

#### Cosine Similarity — The Distance Metric

Two vectors can be compared using **cosine similarity** — the angle between them, not the absolute distance. This matters because texts of different lengths produce vectors of different magnitudes, but the *direction* is what captures meaning.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

Result range: -1 to 1
  1.0 = identical meaning
  0.0 = unrelated
 -1.0 = opposite meaning (rare in practice; most similarities are 0–1)
```

In practice, production similarity thresholds:
- > 0.85: very high match, almost certainly the right document
- 0.70–0.85: likely relevant, worth including in top-k
- 0.50–0.70: possibly relevant, include if you need more results
- < 0.50: likely irrelevant, filter out

**In short:** Cosine similarity measures the angle between two meaning-vectors. Higher angle = more similar meaning. Use 0.70 as your minimum threshold and tune from there.

#### Multilingual Embeddings

Standard OpenAI embeddings handle multilingual text, but quality degrades outside English **because** most embedding models were trained on English-heavy corpora — which means **in practice** a query in French may not retrieve a French document about the same topic.

For multilingual RAG systems, use dedicated multilingual models:

| Use case | Model | Dimensions | Notes |
|----------|-------|-----------|-------|
| English only | `text-embedding-3-small` | 1,536 | Best price/quality for English |
| Multilingual (cloud) | `multilingual-e5-large` | 1,024 | Via HuggingFace Inference |
| Multilingual (local) | `BGE-M3` | 1,024 | Free, SOTA on MTEB multilingual |
| Cross-lingual retrieval | `LaBSE` | 768 | Optimized for matching across languages |

**The rule:** If your users write in Language A and your documents are in Language B, you need a cross-lingual model — a model where the same meaning in two languages produces nearby vectors.

#### The Embedding Pipeline

Every embedding operation follows the same steps:

```
Input text
    ↓
Tokenize (same tokenizer as used during training)
    ↓
Feed through transformer layers
    ↓
Pool the output (mean-pooling across token embeddings)
    ↓
Normalize to unit length (for cosine similarity)
    ↓
Output: vector of N floats
```

The normalization step is critical. After normalizing, cosine similarity equals dot product — which is faster to compute. Most vector databases default to dot product for this reason.

**In short:** The embedding pipeline tokenizes your text, runs it through a transformer, pools the output into a single vector, and normalizes it. The result is a unit-length vector where dot product equals cosine similarity.

---

### Engineering Layer

```python
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import openai
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str
    tokens_used: int
    latency_ms: float


class EmbeddingProvider:
    """Abstract interface — swap providers without changing application code."""

    def embed(self, text: str) -> EmbeddingResult:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        raise NotImplementedError


class OpenAIEmbedder(EmbeddingProvider):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def embed(self, text: str) -> EmbeddingResult:
        text = text.replace("\n", " ")  # newlines degrade quality
        start = time.monotonic()
        response = self.client.embeddings.create(input=[text], model=self.model)
        latency_ms = (time.monotonic() - start) * 1000
        data = response.data[0]
        return EmbeddingResult(
            vector=data.embedding,
            model=self.model,
            tokens_used=response.usage.total_tokens,
            latency_ms=latency_ms,
        )

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[EmbeddingResult]:
        texts = [t.replace("\n", " ") for t in texts]
        results: list[EmbeddingResult] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            start = time.monotonic()
            response = self.client.embeddings.create(input=batch, model=self.model)
            latency_ms = (time.monotonic() - start) * 1000
            per_item_ms = latency_ms / len(batch)
            for j, data in enumerate(response.data):
                results.append(
                    EmbeddingResult(
                        vector=data.embedding,
                        model=self.model,
                        tokens_used=response.usage.total_tokens // len(batch),
                        latency_ms=per_item_ms,
                    )
                )
        return results


class LocalEmbedder(EmbeddingProvider):
    """sentence-transformers — runs on CPU or GPU, zero API cost."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> EmbeddingResult:
        start = time.monotonic()
        vector = self.model.encode(text, normalize_embeddings=True).tolist()
        latency_ms = (time.monotonic() - start) * 1000
        return EmbeddingResult(
            vector=vector,
            model=self.model_name,
            tokens_used=len(text.split()),  # approximate
            latency_ms=latency_ms,
        )

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[EmbeddingResult]:
        start = time.monotonic()
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        ).tolist()
        latency_ms = (time.monotonic() - start) * 1000
        per_item_ms = latency_ms / len(texts)
        return [
            EmbeddingResult(
                vector=v,
                model=self.model_name,
                tokens_used=len(t.split()),
                latency_ms=per_item_ms,
            )
            for v, t in zip(vectors, texts)
        ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Assumes vectors are already L2-normalized (from embed with normalize=True)."""
    return float(np.dot(a, b))  # dot product == cosine sim for unit vectors
```

The `replace("\n", " ")` call before embedding is non-obvious but important: newlines cause embedding models to treat the text as multiple separate segments internally, which can shift the vector. The `normalize_embeddings=True` flag in sentence-transformers ensures dot product equals cosine similarity downstream — skipping this causes incorrect similarity scores when your vector DB uses dot product.

---

### Architecture Layer

#### Embedding Generation at Scale

```
[Document Store]
      │
      ↓ raw text
[Embedding Service]
      │
      ├── Tokenizer check (count tokens, reject if > model max)
      ├── Batching (100 docs per API call for OpenAI)
      ├── Rate limit handling (3,500 RPM for text-embedding-3-small tier 1)
      ├── Retry with exponential backoff (429 → wait → retry)
      └── Cost tracking ($0.00002/1K tokens for text-embedding-3-small)
      │
      ↓ vectors
[Vector Database]
```

```
Embedding cost at scale:
  text-embedding-3-small: $0.00002 / 1K tokens
  10,000 documents × 400 tokens avg = 4M tokens
  4M tokens × $0.00002 / 1K = $0.08 for initial indexing
  
  Re-embedding 100 changed docs/day:
  100 × 400 tokens = 40K tokens/day = $0.0008/day = ~$0.30/year
```

| Metric | text-embedding-3-small | text-embedding-3-large | BGE-M3 (local) |
|--------|----------------------|----------------------|----------------|
| Dimensions | 1,536 | 3,072 | 1,024 |
| Cost/1K tokens | $0.00002 | $0.00013 | $0 |
| Latency (single) | ~100ms | ~120ms | ~50ms CPU / ~5ms GPU |
| MTEB avg score | 62.3 | 64.6 | 54.9 |
| Languages | 100+ | 100+ | 100+ |

> **Bridge:** Now that you can convert text into meaning-vectors, you need to choose which model to use. That choice determines retrieval quality, latency, cost, and multilingual support — and the benchmarks that guide it are specific and measurable.

---

⚡ **Senior Checklist — 3.2**
- [ ] Replace newlines with spaces before embedding — newlines silently degrade vector quality
- [ ] Always normalize embeddings to unit length — without normalization, dot product ≠ cosine similarity
- [ ] Batch embedding calls: OpenAI allows 100 texts per request, reducing latency by ~80× vs one-at-a-time
- [ ] Never mix embedding models across the same index — a vector from model A is incompatible with a query from model B
- [ ] Track embedding cost separately in your cost monitoring: large re-indexing jobs can spike unexpectedly
- [ ] For multilingual corpora, test retrieval on minority languages before committing to a model — MTEB averages hide per-language gaps
- [ ] Store the model name alongside every embedding in your DB — you will need it when you upgrade the model

---

## 3.3 Embedding Model Selection

### 🧠 Mental Model
> MTEB is the leaderboard, but it measures average academic benchmark performance — not your specific domain or language. The model that ranks highest on MTEB may underperform a smaller model fine-tuned on your domain. Always run your own evaluation on 200 representative queries before choosing.

**Connects to:** Embeddings (3.2) → **Embedding Model Selection** → Vector Search (3.4)
**Parent concept:** Model evaluation and selection
**Builds on:** Section 3.2 — you understand what embeddings are; now you need to choose which model produces the best ones for your use case

---

### Concept Layer

#### What MTEB Is

MTEB (Massive Text Embedding Benchmark) is the standard leaderboard for embedding models. It evaluates models across 56 datasets covering 8 task types: retrieval, clustering, classification, reranking, semantic similarity, summarization, cross-lingual alignment (bitext mining), and pair classification.

The number you care about most is the **retrieval score** — specifically the NDCG@10. It measures how well the top-10 results are ranked: did the most relevant document come first? For a RAG system, this is the closest proxy to production retrieval quality.

MTEB scores for major models (as of early 2025):

| Model | Retrieval NDCG@10 | Dims | Params | Provider |
|-------|------------------|------|--------|----------|
| `text-embedding-3-large` | 59.2 | 3,072 | — | OpenAI |
| `text-embedding-3-small` | 54.9 | 1,536 | — | OpenAI |
| `E5-mistral-7b-instruct` | 56.9 | 4,096 | 7B | HuggingFace |
| `BGE-large-en-v1.5` | 54.3 | 1,024 | 335M | BAAI |
| `BGE-M3` | 55.6 | 1,024 | 570M | BAAI |
| `Cohere embed-english-v3.0` | 55.0 | 1,024 | — | Cohere |

**The rule:** Do not pick the highest MTEB number and stop. MTEB is a proxy. Your domain data is the ground truth. A 2-point MTEB gap does not predict a 2-point retrieval gap on your data.

#### The Four Dimensions of Selection

**1. Quality (MTEB retrieval score)**
Higher is better, but differences below 2 points are often within noise margin on your specific data.

**2. Latency**
Cloud models: 80–150ms per batch API call (regardless of batch size, up to the limit).
Local models: 5ms on A100 GPU, 50ms on modern CPU per document.

**3. Cost**
- `text-embedding-3-small`: $0.00002/1K tokens → $0.08 to embed 10K documents
- `text-embedding-3-large`: $0.00013/1K tokens → $0.52 to embed 10K documents
- Local models: GPU electricity cost ≈ $0.002/hour on A100, effectively zero at scale

**4. Dimension tradeoffs**
Higher dimensions capture more semantic nuance but cost more storage and slow down search.

```
Storage per 1M vectors:
  768 dims  × 4 bytes × 1M = 3 GB
  1,024 dims × 4 bytes × 1M = 4 GB
  1,536 dims × 4 bytes × 1M = 6 GB
  3,072 dims × 4 bytes × 1M = 12 GB
```

For most production systems: 1,024–1,536 dimensions is the sweet spot. Beyond 1,536, quality gains are marginal and storage costs double.

**In short:** Pick by retrieval NDCG@10 first, then filter by latency and cost constraints, then validate on your actual domain data before committing.

#### Matryoshka Embeddings

OpenAI's `text-embedding-3-*` models use Matryoshka Representation Learning (MRL). You can **truncate the vector to fewer dimensions** without a full quality loss:

```
text-embedding-3-small (full 1536 dims): MTEB 62.3
text-embedding-3-small (truncated to 512 dims): MTEB 61.0  ← 2% quality drop, 67% storage saving
text-embedding-3-small (truncated to 256 dims): MTEB 60.1  ← 3% quality drop, 83% storage saving
```

This means you can use smaller vectors for first-pass retrieval (faster, cheaper) and full vectors for reranking — a common production optimization.

**The rule:** Use Matryoshka truncation to 512 dimensions for your first-pass ANN index, then rerank with full vectors if needed. You get 67% storage savings with only 2% quality loss.

---

### Engineering Layer

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class ModelBenchmarkResult:
    model_name: str
    avg_similarity_relevant: float
    avg_similarity_irrelevant: float
    separation: float  # the gap — larger is better
    avg_latency_ms: float
    p95_latency_ms: float


def benchmark_embedding_model(
    model_name: str,
    query_doc_pairs: list[tuple[str, str]],  # (query, relevant_doc) pairs
    negative_docs: list[str],
    device: str = "cpu",
    runs: int = 3,
) -> ModelBenchmarkResult:
    """
    Evaluate an embedding model on your actual domain data.
    query_doc_pairs: list of (query, known-relevant-document) pairs
    negative_docs: documents known to be irrelevant to the queries
    """
    model = SentenceTransformer(model_name, device=device)

    queries = [q for q, _ in query_doc_pairs]
    relevant_docs = [d for _, d in query_doc_pairs]
    all_docs = relevant_docs + negative_docs

    latencies: list[float] = []
    for _ in range(runs):
        start = time.monotonic()
        q_embs = model.encode(queries, normalize_embeddings=True)
        doc_embs = model.encode(all_docs, normalize_embeddings=True)
        latencies.append((time.monotonic() - start) * 1000)

    q_embs = model.encode(queries, normalize_embeddings=True)
    doc_embs = model.encode(all_docs, normalize_embeddings=True)
    rel_embs = doc_embs[: len(relevant_docs)]
    neg_embs = doc_embs[len(relevant_docs) :]

    relevant_sims = [float(np.dot(q, d)) for q, d in zip(q_embs, rel_embs)]
    irrelevant_sims = [
        float(np.dot(q, d))
        for q in q_embs
        for d in neg_embs
    ]

    avg_rel = float(np.mean(relevant_sims))
    avg_irr = float(np.mean(irrelevant_sims))
    latency_arr = np.array(latencies)

    return ModelBenchmarkResult(
        model_name=model_name,
        avg_similarity_relevant=avg_rel,
        avg_similarity_irrelevant=avg_irr,
        separation=avg_rel - avg_irr,
        avg_latency_ms=float(np.mean(latency_arr)),
        p95_latency_ms=float(np.percentile(latency_arr, 95)),
    )


def select_best_model(
    results: list[ModelBenchmarkResult],
    max_latency_ms: float = 100.0,
) -> ModelBenchmarkResult:
    eligible = [r for r in results if r.p95_latency_ms <= max_latency_ms]
    if not eligible:
        eligible = results  # relax constraint if nothing fits
    return max(eligible, key=lambda r: r.separation)
```

The key metric here is `separation` — the gap between how similar the model scores relevant vs irrelevant documents. A model with high MTEB but low separation on your domain data is a poor fit. This benchmark runs in under 5 minutes on 200 query pairs and gives you the ground-truth answer the leaderboard cannot.

---

### Architecture Layer

#### Model Selection Decision Flow

```
START: Choose embedding model for production RAG
      │
      ├─ English only, cloud OK, budget flexible?
      │         └──→ text-embedding-3-large ($0.00013/1K tokens)
      │
      ├─ English only, cost-sensitive?
      │         └──→ text-embedding-3-small ($0.00002/1K tokens)
      │
      ├─ Multilingual (5+ languages)?
      │         ├── No GPU available → text-embedding-3-small (handles multilingual, lower quality)
      │         └── GPU available   → BGE-M3 (local, SOTA multilingual, 0 cost)
      │
      ├─ Data privacy (no third-party APIs)?
      │         └──→ BGE-large-en-v1.5 or BGE-M3 (local, self-hosted)
      │
      └─ High query volume (>1M/day)?
                └──→ Local model on GPU cluster (API cost > GPU cost at that scale)
                     Break-even: ~500K queries/day at text-embedding-3-small pricing
```

| Constraint | Recommended Model | Reason |
|-----------|------------------|--------|
| SLA < 50ms | Local BGE-M3 on GPU | Cloud APIs add 80–150ms network latency |
| GDPR / no data egress | BGE-large-en-v1.5 (local) | Data never leaves your infrastructure |
| Storage < 4 GB / 1M vectors | BGE-M3 (1,024 dims) | 4 GB vs 6 GB for 1,536-dim models |
| Matryoshka truncation needed | text-embedding-3-small/large | Only OpenAI models support native truncation |

> **Bridge:** You now know how to turn documents into vectors and which model to use. The next question is: given a million vectors, how do you find the 5 most similar ones in under 50 milliseconds? That is the vector search problem.

---

⚡ **Senior Checklist — 3.3**
- [ ] Run your own domain benchmark before choosing a model — MTEB scores are averages over academic datasets, not your data
- [ ] Measure `separation` (relevant similarity minus irrelevant similarity), not just absolute similarity scores
- [ ] At >500K queries/day, local GPU hosting costs less than cloud embedding APIs — calculate break-even before committing
- [ ] Use Matryoshka truncation to 512 dimensions for first-pass retrieval when storage costs matter: 67% savings, 2% quality loss
- [ ] Never mix embedding models in one index — upgrade requires full re-embedding of every document
- [ ] Store model name and version in your vector DB metadata schema — you will forget which model produced which index
- [ ] Test p95 latency under batch load, not just median latency — spikes determine SLA compliance

---

## 3.4 Vector Search Basics

### 🧠 Mental Model
> Exact nearest-neighbor search is like calling every phone number in the country to find the right person. Approximate nearest-neighbor (ANN) search is like looking up in a phone book — you skip 99% of candidates and still find the right answer 95–99% of the time. At a million vectors, ANN is not a compromise; it is the only viable option.

**Connects to:** Embedding Model Selection (3.3) → **Vector Search** → Vector DB Selection (3.5)
**Parent concept:** Information retrieval algorithms
**Builds on:** Section 3.2 — you can compute cosine similarity between two vectors; vector search is computing it efficiently across millions

---

### Concept Layer

#### Exact vs Approximate Search

You have 1 million document vectors and a query vector. You want the 5 most similar documents.

**Exact search (brute force):** Compare the query against all 1 million vectors. At 1,536 dimensions, that is 1,536 × 1,000,000 = 1.5 billion floating-point multiplications. On a modern CPU: ~2 seconds. On a GPU: ~50ms. Fine for 100K vectors, unacceptable at 10M.

**Approximate Nearest Neighbor (ANN):** Build an index that skips most comparisons. Trade a small quality loss (missing 1–5% of true nearest neighbors) for a large speed gain (10–100× faster). At production scale, ANN is not optional.

**The rule:** Use exact search for fewer than 50K vectors (it fits in RAM and is fast enough). Use ANN for anything above that.

#### HNSW — The Production Standard

HNSW (Hierarchical Navigable Small World) is the dominant ANN algorithm. It builds a multi-layer graph where each vector is a node and edges connect similar vectors. Search navigates the graph from a coarse top layer down to a fine bottom layer.

How HNSW search works:

```
Layer 2 (sparse, long-range connections):
  Start ──→ hop to closest node ──→ hop again ──→ reach approximate region

Layer 1 (denser):
  Refine within the region

Layer 0 (full density, all connections):
  Exact search within the final candidate set
```

Key parameters:
- `M` (max connections per node): 16–64. Higher = better recall, more memory. Use 16 for standard, 32 for high-recall.
- `ef_construction` (candidates during build): 64–200. Higher = better index quality, slower build. Use 128 for balanced, 200 for best quality.
- `ef_search` (candidates during query): 50–200. Higher = better recall, slower query. Tune at runtime without rebuilding.

**HNSW properties:**
- Build time: O(n × M × log n) — slow to build, fast to query
- Search time: O(log n) — logarithmic in the number of vectors
- Memory: ~(M × dim × 4 bytes) per vector → at M=16, 1,536 dims: ~100 KB per vector → ~100 GB per million vectors (rough estimate)
- Recall at ef_search=200: 97–99% (misses 1–3% of true nearest neighbors)

#### IVFFlat — The Alternative

IVFFlat (Inverted File with Flat quantization) works differently: it clusters vectors into `nlist` buckets (e.g., 1,024 clusters using k-means), then at search time searches only `nprobe` clusters (e.g., 10).

```
Build: cluster 1M vectors into 1,024 groups
Query: find the 10 nearest cluster centroids
       search all vectors in those 10 clusters (~10K vectors)
       return top-k from that subset
```

IVFFlat is faster to build than HNSW but slower to query at the same recall. It requires a training step (the k-means clustering) before inserting vectors. It works best when the number of vectors is 10× larger than `nlist`.

| Property | HNSW | IVFFlat |
|----------|------|---------|
| Build speed | Slow (minutes for 1M) | Fast (seconds for clustering) |
| Query speed | Fast (O(log n)) | Fast (O(nprobe × cluster_size)) |
| Recall at default settings | 97–99% | 90–95% |
| Memory usage | High (~100 GB/1M 1536-dim) | Lower (only centroids + inverted lists) |
| Online insertion | Yes (no rebuild) | Requires rebuild for new vectors |
| Best for | Dynamic indexes, high recall | Static bulk indexes, memory-constrained |

**The rule:** Use HNSW as your default. Switch to IVFFlat only if your index is static (no frequent inserts) and memory is constrained.

**In short:** HNSW searches like navigating a road network — fast because you skip dead ends. IVFFlat searches like checking the 10 nearest neighborhoods — fast because you ignore the rest of the city. HNSW is better for dynamic indexes; IVFFlat is better for static bulk data.

#### Dot Product vs Cosine Similarity vs Euclidean Distance

```
Cosine similarity: measures the angle between vectors (ignores magnitude)
Dot product: cosine similarity × magnitude × magnitude
Euclidean distance: straight-line distance in high-dimensional space

When vectors are L2-normalized (unit length):
  dot product == cosine similarity
  → use dot product (faster computation, same result)

When vectors are NOT normalized:
  use cosine similarity explicitly
  → do NOT use dot product (magnitude differences contaminate results)
```

**The rule:** Always normalize your embeddings to unit length. Then use dot product. It gives you cosine similarity at lower computational cost.

---

### Engineering Layer

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import hnswlib
import numpy as np


@dataclass
class SearchResult:
    id: int
    score: float
    metadata: dict[str, Any]


class HNSWIndex:
    """
    In-memory HNSW index using hnswlib.
    For production, use pgvector or Qdrant which manage persistence.
    This example shows the raw algorithm so you understand what happens inside.
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 100_000,
        M: int = 16,
        ef_construction: int = 200,
        metric: str = "cosine",  # "cosine" or "l2" or "ip" (inner product)
    ) -> None:
        self.dim = dim
        self.index = hnswlib.Index(space=metric, dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=42,
        )
        self.index.set_ef(50)  # ef_search default — can tune at query time
        self._metadata: dict[int, dict[str, Any]] = {}

    def add(self, vectors: np.ndarray, ids: list[int], metadata: list[dict[str, Any]]) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        # Normalize for cosine — critical if embedding model doesn't normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1, norms)
        self.index.add_items(vectors, ids)
        for id_, meta in zip(ids, metadata):
            self._metadata[id_] = meta

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        ef_search: int | None = None,
    ) -> list[SearchResult]:
        if ef_search is not None:
            self.index.set_ef(ef_search)

        query = query / np.linalg.norm(query)  # normalize query too
        labels, distances = self.index.knn_query(query.reshape(1, -1), k=k)

        results = []
        for label, dist in zip(labels[0], distances[0]):
            # hnswlib returns L2 distance for cosine space — convert to similarity
            score = 1.0 - (dist / 2.0)  # for cosine metric in hnswlib
            results.append(
                SearchResult(
                    id=int(label),
                    score=score,
                    metadata=self._metadata.get(int(label), {}),
                )
            )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def save(self, path: str) -> None:
        self.index.save_index(path)

    @classmethod
    def load(cls, path: str, dim: int, max_elements: int = 100_000) -> "HNSWIndex":
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.index = hnswlib.Index(space="cosine", dim=dim)
        obj.index.load_index(path, max_elements=max_elements)
        obj._metadata = {}
        return obj
```

The normalization inside `add()` is defensive: it normalizes even if the embedding provider claims to have already done it. A missing normalization step is the most common cause of "the search returns wrong results" bugs, and it is nearly impossible to detect without examining raw vector norms. The `ef_search` override allows you to trade recall for speed at query time without rebuilding the index — useful for bulk evaluation runs.

---

### Architecture Layer

#### Vector Search in a RAG System

```
[Query text]
      │
      ↓ embed (50–150ms)
[Query vector]
      │
      ↓ ANN search (2–20ms with HNSW)
[Top-k candidate IDs] ──→ [Metadata store] ──→ [Filtered candidates]
      │                         │
      │                    (apply metadata filters:
      │                     tenant, date, access level)
      ↓
[Re-fetch full chunk text from document store]
      │
      ↓
[Assemble context for LLM prompt]
```

#### Recall vs Latency Tradeoff

```
ef_search | Recall (%) | Latency (ms, 1M vectors)
──────────┼────────────┼─────────────────────────
    10    |    85%     |     2ms
    50    |    93%     |     5ms
   100    |    96%     |     9ms
   200    |    98%     |    18ms
   400    |    99%     |    35ms

M=16, ef_construction=200, 1M 1536-dim vectors, single thread
```

For most production RAG systems: `ef_search=100` gives 96% recall at 9ms — a good default. Raise to 200 if your evaluation shows retrieval misses are the dominant failure mode.

> **Bridge:** You understand how ANN search finds similar vectors at scale. The next decision is which system to actually store and search those vectors in production — and the choice depends on deployment constraints, not just search algorithm quality.

---

⚡ **Senior Checklist — 3.4**
- [ ] Normalize all vectors to unit length before indexing — missing this causes silent wrong results with dot-product metric
- [ ] Set `ef_search` to 100 as your starting default — 96% recall at 9ms for 1M vectors
- [ ] Use HNSW for dynamic indexes (frequent inserts), IVFFlat for static bulk indexes
- [ ] Benchmark recall on your actual data: generate 200 query-document pairs, measure how often true match appears in top-k
- [ ] Set `M=16, ef_construction=200` for standard indexes; raise `M=32` only when recall benchmarks show misses
- [ ] Account for HNSW memory: 100 GB for 1M 1536-dim vectors at M=16 — size your instance accordingly
- [ ] Never use Euclidean distance for text embeddings — cosine similarity is semantically correct; Euclidean mixes magnitude with direction

---

## 3.5 Vector Database Selection

### 🧠 Mental Model
> pgvector is your Postgres column. Qdrant is a dedicated search engine. Pinecone is fully managed search. Weaviate is search plus object storage. The right choice is not about which has the best algorithm — it is about which fits your existing stack, team skills, and operational constraints.

**Connects to:** Vector Search (3.4) → **Vector DB Selection** → Document Ingestion (3.6)
**Parent concept:** Infrastructure selection for RAG
**Builds on:** Section 3.4 — HNSW and IVFFlat are the algorithms; pgvector, Qdrant, Pinecone, and Weaviate are the systems that implement them

---

### Concept Layer

#### The Four Contenders

**pgvector** adds a `vector` column type to PostgreSQL. It implements HNSW and IVFFlat inside Postgres. Your vectors live in the same database as your application data.

Use pgvector when:
- Your team already runs Postgres
- You need SQL JOINs between vectors and relational data
- Your scale is under 1M vectors (pgvector degrades above this without careful tuning)
- You want zero new infrastructure

**Qdrant** is a purpose-built vector database written in Rust. It implements HNSW with payload filtering (you can filter by metadata at search time, not post-search). It runs as a standalone service.

Use Qdrant when:
- You have >1M vectors
- You need payload-level filtering (tenant isolation, date filtering)
- You want self-hosted control with production-grade features
- You need quantization (int8, binary) to reduce memory

**Pinecone** is a fully managed vector database. You pay per pod. No server management.

Use Pinecone when:
- Your team cannot manage infrastructure
- You need to ship in days, not weeks
- You can accept vendor lock-in for operational simplicity
- Budget allows ~$70–$200/month for a production pod

**Weaviate** combines a vector index with object storage. Vectors and full document content live together. It includes a built-in BM25 index for hybrid search.

Use Weaviate when:
- You want hybrid search (BM25 + dense) out of the box
- You prefer GraphQL query interface
- You need vector + full-text in one service

**In short:** pgvector for teams already running Postgres at moderate scale. Qdrant for production self-hosted at large scale. Pinecone when you want managed infrastructure. Weaviate when you want hybrid search built in.

#### The Scale Thresholds

These are rough guidelines from production deployments:

| Vectors | Recommendation | Why |
|---------|---------------|-----|
| < 100K | pgvector, no index needed | Exact search is fast enough |
| 100K–1M | pgvector with HNSW or Qdrant | Either works; pgvector is simpler |
| 1M–10M | Qdrant or Weaviate | pgvector needs sharding at this scale |
| > 10M | Qdrant (distributed), Pinecone | Dedicated DB designed for this |

---

### Engineering Layer

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, Any]
    text: str


# ── pgvector ─────────────────────────────────────────────────────────────────

class PgVectorStore:
    """Requires: pip install psycopg2-binary pgvector"""

    def __init__(self, connection_string: str, table: str = "documents") -> None:
        import psycopg2
        from pgvector.psycopg2 import register_vector
        self.conn = psycopg2.connect(connection_string)
        register_vector(self.conn)
        self.table = table
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id          TEXT PRIMARY KEY,
                    tenant_id   TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    metadata    JSONB,
                    embedding   vector(1536)
                );
                CREATE INDEX IF NOT EXISTS {self.table}_hnsw_idx
                ON {self.table}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200);
            """)
        self.conn.commit()

    def upsert(self, id_: str, text: str, embedding: list[float],
               tenant_id: str, metadata: dict[str, Any]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.table} (id, tenant_id, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET text=EXCLUDED.text, metadata=EXCLUDED.metadata,
                    embedding=EXCLUDED.embedding
            """, (id_, tenant_id, text, metadata, embedding))
        self.conn.commit()

    def search(self, query_embedding: list[float], tenant_id: str,
               k: int = 5, min_score: float = 0.70) -> list[VectorSearchResult]:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.table}
                WHERE tenant_id = %s
                  AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, tenant_id, query_embedding, min_score, query_embedding, k))
            rows = cur.fetchall()
        return [
            VectorSearchResult(id=r[0], text=r[1], payload=r[2] or {}, score=float(r[3]))
            for r in rows
        ]


# ── Qdrant ────────────────────────────────────────────────────────────────────

class QdrantStore:
    """Requires: pip install qdrant-client"""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = "documents",
        dim: int = 1536,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection

        existing = [c.name for c in self.client.get_collections().collections]
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, id_: str, text: str, embedding: list[float],
               tenant_id: str, metadata: dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct
        import hashlib
        # Qdrant requires integer IDs — hash the string ID
        int_id = int(hashlib.md5(id_.encode()).hexdigest()[:8], 16)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=int_id,
                vector=embedding,
                payload={"text": text, "doc_id": id_, "tenant_id": tenant_id, **metadata},
            )],
        )

    def search(self, query_embedding: list[float], tenant_id: str,
               k: int = 5, min_score: float = 0.70) -> list[VectorSearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            ),
            limit=k,
            score_threshold=min_score,
            with_payload=True,
        )
        return [
            VectorSearchResult(
                id=r.payload["doc_id"],
                score=r.score,
                payload={k: v for k, v in r.payload.items() if k not in ("text", "doc_id")},
                text=r.payload["text"],
            )
            for r in results
        ]
```

The pgvector `<=>` operator is cosine distance (1 - cosine similarity). Converting to similarity (`1 - distance`) before returning results keeps the interface consistent with Qdrant, which returns similarity directly. The MD5 hash for Qdrant IDs is a workaround for Qdrant's integer-only ID requirement — store the original string ID in the payload.

---

### Architecture Layer

#### Deployment Topology

```
pgvector deployment:
  [App] ──→ [Postgres + pgvector extension]
             (same instance as app DB, or separate)
             Managed: AWS RDS, Supabase, Neon

Qdrant deployment:
  [App] ──→ [Qdrant service] (Docker or Kubernetes)
  [App] ──→ [Qdrant Cloud] (managed, ~$0.12/hour)

Pinecone deployment:
  [App] ──→ [Pinecone API] (fully managed, no servers)
             Starter: free, 100K vectors
             Standard: $0.096/hour per p1.x1 pod (1M vectors)
```

| Factor | pgvector | Qdrant | Pinecone | Weaviate |
|--------|----------|--------|----------|---------|
| Setup time | Minutes (extension) | 30 min (Docker) | 5 min (API key) | 30 min |
| Max scale | ~5M vectors tuned | 100M+ | 1B+ | 100M+ |
| Self-hosted | Yes (it IS Postgres) | Yes | No | Yes |
| Payload filtering | Post-query (SQL WHERE) | At index time | At query time | At query time |
| Hybrid search | No (add pg_bm25) | Sparse + dense | No (need workaround) | Built-in |
| Monthly cost (1M vectors) | $0 (if Postgres exists) | $80 (cloud) | $70–$200 | $85 (cloud) |

> **Bridge:** You have chosen where to store your vectors. Before you can search them, you need to get your documents in — and that means building an ingestion pipeline that handles the messy reality of PDFs, HTML pages, Word documents, and Markdown files.

---

⚡ **Senior Checklist — 3.5**
- [ ] Start with pgvector if you already run Postgres — adds zero new operational complexity below 1M vectors
- [ ] Use Qdrant's payload filtering, not post-search Python filtering — filtering at index time is 10–100× faster
- [ ] Set Qdrant quantization (int8) at >5M vectors — reduces RAM by 4×, minimal quality loss
- [ ] Test Pinecone pod sizing before committing: p1.x1 holds 1M 1536-dim vectors, p1.x2 holds 2M
- [ ] Never store full document text in the vector DB — store chunk IDs and retrieve text from your document store
- [ ] Add a `schema_version` field to every vector payload — you will migrate schemas when you upgrade embedding models
- [ ] Run regular recall benchmarks against your vector DB, not just the ANN algorithm — database-level filtering can reduce effective recall

---

## 3.6 Document Ingestion Pipelines

### 🧠 Mental Model
> Every document format is a container for text plus noise. Your ingestion pipeline's job is to extract the text and discard the noise — formatting artifacts, headers/footers, navigation menus, binary content — before the text reaches the chunker. Garbage in, garbage chunks out.

**Connects to:** Vector DB Selection (3.5) → **Document Ingestion** → Chunking (3.7)
**Parent concept:** Data preprocessing for RAG
**Builds on:** Section 3.5 — you have a vector store; now you need to fill it with clean document chunks

---

### Concept Layer

#### Why Ingestion Is Not Trivial

A PDF that looks clean on screen contains a stream of positioning commands, font codes, and character codes — not sentences. Extracting the sentence "The refund window is 30 days" from a PDF often produces "The refund window is30 days" (missing space) or "The refund\nwindow is 30 days" (line break in the middle). These artifacts break chunking and degrade retrieval.

HTML pages contain navigation menus, cookie banners, advertisement blocks, and footer links — all of which look like text to an extractor. Chunking this produces vectors that point to "Accept All Cookies" and "© 2024 Company Name" — legitimate text, useless retrieval targets.

Word documents (DOCX) use XML under the hood. Tables render as flat text with no structure. Tracked changes add deleted text mixed with current text if not stripped.

**The rule:** Every format requires a format-specific extraction step. A generic "extract all text" approach produces a retrieval system that returns cookie banners and PDF formatting artifacts.

#### The Four Format Strategies

**PDF:**
- Use `pdfplumber` for text-heavy PDFs with predictable layouts
- Use `unstructured` for PDFs with mixed layouts, tables, figures
- Use `pytesseract` + `pdf2image` for scanned PDFs (image-based OCR)
- Extract page numbers — they become metadata for citations

**HTML:**
- Use `BeautifulSoup` to strip navigation, footers, scripts, styles
- Keep `<h1>`–`<h3>` tags as section context (discard the tag, keep the text as a heading signal)
- `readability-lxml` extracts main article content automatically (like "Reader Mode" in browsers)

**DOCX:**
- Use `python-docx` to access paragraph objects directly
- Strip tracked changes: only process `document.paragraphs`, which returns accepted-state text
- Preserve heading levels as metadata for chunk context

**Markdown:**
- Split on `##` headers to preserve section context
- Strip frontmatter (YAML between `---` delimiters)
- Keep code blocks intact — split around them, never through them

**In short:** Extract format-specific structure first (section headings, page numbers, table context), then clean the text, then hand it to the chunker. Never hand raw format output to the chunker.

#### The Ingestion Quality Checklist

Before a chunk reaches the vector store, verify:

1. No more than 5% of characters are non-printable
2. No lines shorter than 3 words (these are usually layout artifacts)
3. No repeated strings (headers/footers copied to every page)
4. Text is UTF-8 — re-encode it if it comes in as Latin-1 or Windows-1252

---

### Engineering Layer

```python
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ExtractedDocument:
    source_path: str
    format: str
    pages: list[str]             # one string per page/section
    headings: list[str]          # section headings in order
    metadata: dict               # title, author, page_count, etc.
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        combined = "\n".join(self.pages)
        self.content_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract text from PDF, one string per page."""
    import pdfplumber

    pages: list[str] = []
    metadata: dict = {}

    with pdfplumber.open(path) as pdf:
        metadata["page_count"] = len(pdf.pages)
        metadata["title"] = pdf.metadata.get("Title", path.stem)

        for page in pdf.pages:
            text = page.extract_text() or ""
            text = _clean_pdf_text(text)
            if len(text.strip()) > 50:  # skip nearly-empty pages
                pages.append(text)

    return ExtractedDocument(
        source_path=str(path),
        format="pdf",
        pages=pages,
        headings=[],  # pdfplumber doesn't extract heading structure
        metadata=metadata,
    )


def _clean_pdf_text(text: str) -> str:
    """Remove common PDF extraction artifacts."""
    # Fix missing spaces before numbers (common PDF artifact)
    text = re.sub(r"(\w)(\d)", r"\1 \2", text)
    # Collapse multiple whitespace, preserve single newlines
    text = re.sub(r" {2,}", " ", text)
    # Remove lines that are only punctuation or single characters (page numbers, bullets)
    lines = [l for l in text.splitlines() if len(l.strip()) > 2]
    return "\n".join(lines)


def extract_html(path: Path) -> ExtractedDocument:
    """Extract main content from HTML, stripping nav/footer/ads."""
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # Remove boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    title = soup.title.string if soup.title else path.stem

    # Extract paragraphs and headings in reading order
    content_parts: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = tag.get_text(" ", strip=True)
        if len(text) > 20:  # skip trivial snippets
            content_parts.append(text)

    return ExtractedDocument(
        source_path=str(path),
        format="html",
        pages=["\n".join(content_parts)],
        headings=headings,
        metadata={"title": title},
    )


def extract_docx(path: Path) -> ExtractedDocument:
    """Extract paragraphs and headings from DOCX."""
    import docx

    doc = docx.Document(path)
    pages: list[str] = []
    headings: list[str] = []
    current_section: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            if current_section:
                pages.append("\n".join(current_section))
                current_section = []
            headings.append(text)
            current_section.append(f"## {text}")
        else:
            current_section.append(text)

    if current_section:
        pages.append("\n".join(current_section))

    props = doc.core_properties
    return ExtractedDocument(
        source_path=str(path),
        format="docx",
        pages=pages,
        headings=headings,
        metadata={"title": props.title or path.stem, "author": props.author or ""},
    )


def extract_markdown(path: Path) -> ExtractedDocument:
    """Split Markdown by H2 sections, strip frontmatter."""
    text = path.read_text(encoding="utf-8")

    # Strip YAML frontmatter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].lstrip()

    # Split on H2 headers
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    pages = [f"## {s}".strip() if i > 0 else s.strip() for i, s in enumerate(sections)]
    pages = [p for p in pages if len(p) > 50]

    headings = re.findall(r"^#{1,3} (.+)$", text, re.MULTILINE)

    return ExtractedDocument(
        source_path=str(path),
        format="markdown",
        pages=pages,
        headings=headings,
        metadata={"title": path.stem},
    )


def extract_document(path: Path) -> ExtractedDocument:
    """Route to the correct extractor by file extension."""
    extractors = {
        ".pdf": extract_pdf,
        ".html": extract_html,
        ".htm": extract_html,
        ".docx": extract_docx,
        ".md": extract_markdown,
        ".txt": lambda p: ExtractedDocument(
            source_path=str(p),
            format="txt",
            pages=[p.read_text(encoding="utf-8", errors="replace")],
            headings=[],
            metadata={"title": p.stem},
        ),
    }
    suffix = path.suffix.lower()
    extractor = extractors.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported format: {suffix}")
    return extractor(path)
```

The `_clean_pdf_text` function fixes the most common PDF artifact — missing spaces — which would otherwise produce nonsense tokens during embedding. Splitting DOCX on heading boundaries (rather than page boundaries) preserves semantic sections, which improves chunking quality downstream.

---

### Architecture Layer

#### Ingestion Pipeline Architecture

```
[Document Store] (S3, GCS, local filesystem)
      │
      ↓ new file event (S3 notification / file watcher)
[Ingestion Queue] (SQS, Redis, Celery)
      │
      ├─ PDF worker ──→ pdfplumber / unstructured / OCR
      ├─ HTML worker ──→ BeautifulSoup / readability
      ├─ DOCX worker ──→ python-docx
      └─ MD worker   ──→ regex splitter
      │
      ↓ ExtractedDocument
[Quality Filter]
      ├── min_length check
      ├── encoding check
      └── duplicate check (SHA-256 content hash)
      │
      ↓ clean text
[Chunker] ──→ [Embedder] ──→ [Vector Store]
                                    │
                              [Metadata Store]
```

**Deduplication strategy:** Hash the content (`SHA-256`) before chunking. If the hash already exists in your metadata store, skip re-embedding. This prevents duplicate chunks from re-indexed documents from polluting retrieval.

| Format | Recommended Library | OCR fallback | Avg extraction time |
|--------|-------------------|--------------|-------------------|
| PDF (text) | `pdfplumber` | — | 0.5s per page |
| PDF (scanned) | `unstructured` + Tesseract | `pytesseract` | 3–10s per page |
| HTML | `BeautifulSoup4` + `readability-lxml` | — | <0.1s |
| DOCX | `python-docx` | — | 0.2s per file |
| Markdown | stdlib `re` | — | <0.01s |

> **Bridge:** You have clean extracted text from each document. Now you need to decide how to split it into chunks — because a 20-page PDF cannot go into one vector, and how you split determines what you can retrieve.

---

⚡ **Senior Checklist — 3.6**
- [ ] Hash document content (SHA-256) at ingestion and skip re-processing unchanged files — re-indexing entire corpora on every run is expensive and wasteful
- [ ] Use OCR for scanned PDFs — text-based PDF extraction on image PDFs returns empty strings silently
- [ ] Strip navigation and footer content from HTML before chunking — these produce high-frequency vectors that pollute retrieval results
- [ ] Preserve section headings as chunk metadata, not just as text — they become citation context later
- [ ] Test your extractor on 10 real documents from your corpus before building the pipeline — format quirks are always worse than expected
- [ ] Route extraction to workers by format type — PDF OCR jobs take 10× longer than Markdown parsing and should not block the queue
- [ ] Store `source_path`, `format`, and `content_hash` with every chunk — you need them for freshness tracking and deduplication

---

## 3.7 Chunking Strategies

### 🧠 Mental Model
> A chunk is the unit of retrieval. If your chunk is too large, it contains irrelevant text that dilutes the embedding. If it is too small, it lacks the context needed for the model to answer. The right chunk size is the smallest unit that can stand alone and answer a question.

**Connects to:** Document Ingestion (3.6) → **Chunking** → Metadata Design (3.8)
**Parent concept:** Text segmentation for retrieval
**Builds on:** Section 3.6 — you have clean extracted text; chunking decides how to divide it into retrievable units

---

### Concept Layer

#### Why Chunking Matters More Than Most Engineers Expect

A naive approach: split every document into 500-token chunks with no overlap. This breaks sentences at boundaries, splits related ideas across chunks, and produces chunks that contain half of one concept and half of another. The embedding of such a chunk points to neither concept reliably.

The impact is concrete: a chunk containing "refund window is 30 days. Subscription renewal happens on the billing date." produces a vector that is equidistant between refund queries and billing queries — it is a poor match for either.

**The rule:** A chunk should answer exactly one question type. If a chunk answers two different questions, split it.

#### Strategy 1: Fixed-Size Chunking

Split every N tokens with M tokens of overlap between adjacent chunks.

```
Document: "The refund window is 30 days. Subscription renewal happens on the billing date. Cancellation takes effect at the end of the billing cycle."

chunk_size=50, overlap=10:
Chunk 1: "The refund window is 30 days. Subscription renewal happens on the billing"
Chunk 2: "happens on the billing date. Cancellation takes effect at the end of"
Chunk 3: "at the end of the billing cycle."
```

Overlap prevents information loss at boundaries — "billing date" appears in both chunk 1 and chunk 2, so a query about billing dates will match at least one chunk completely.

**When to use:** Homogeneous documents (all plain prose, no sections). Simple to implement, predictable size, easy to tune.
**Problem:** Ignores sentence boundaries. Frequently splits mid-sentence, producing incoherent chunks.

#### Strategy 2: Recursive Character Splitting

Split on `\n\n` first (paragraphs), then `\n` (lines), then `. ` (sentences), then ` ` (words) — whichever produces chunks near the target size without splitting on a boundary more granular than necessary.

This is the LangChain `RecursiveCharacterTextSplitter` approach. It respects paragraph and sentence structure **because** text is naturally nested — paragraphs hold sentences, sentences hold words. **In practice** this means splits happen at the cleanest available boundary, not mid-sentence.

**When to use:** General-purpose for most text. Better than fixed-size for prose documents.
**Problem:** Still ignores semantic content — a long paragraph about two topics gets kept together.

#### Strategy 3: Semantic Chunking

Embed every sentence. Split when the cosine similarity between adjacent sentence embeddings drops below a threshold (e.g., 0.7). This produces chunks that contain semantically coherent content.

```
Sentence 1: "The refund window is 30 days."       → embed
Sentence 2: "Refunds are processed within 5 days." → embed → similarity 0.87 → same chunk
Sentence 3: "To cancel, go to account settings."   → embed → similarity 0.52 → NEW CHUNK
Sentence 4: "Cancellation takes effect immediately." → embed → similarity 0.81 → same chunk
```

**Cost:** Requires embedding every sentence individually during ingestion. Adds 5–20× embedding cost at ingestion time.
**When to use:** High-value documents where retrieval quality matters most — legal contracts, technical documentation, medical records.
**Problem:** Expensive. Sentence embedding cost at ingestion can exceed query embedding cost.

#### Strategy 4: Parent-Child Chunking

Store documents at two granularities: large parent chunks (512–1024 tokens) for retrieval context, small child chunks (128–256 tokens) for precise embedding. At query time: search child chunks, return parent chunks to the model.

```
Parent chunk (1024 tokens):
  "Section 3: Billing and Payments
   The refund window is 30 days from purchase.
   Refunds are processed within 5 business days.
   Subscription renewal happens automatically on your billing date..."

Child chunks (128 tokens each) — used for search:
  Child 1: "The refund window is 30 days from purchase."
  Child 2: "Refunds are processed within 5 business days."
  Child 3: "Subscription renewal happens automatically on your billing date."
```

**Why this works:** Small child chunks produce precise embeddings (one concept = one vector). Large parent chunks give the model enough context to answer well. The model never sees the child chunk alone — it sees the full parent context.

**When to use:** Documentation with distinct paragraphs. Enterprise knowledge bases where answer quality per query matters.

**In short:** Fixed-size is simple and predictable. Recursive splitting respects sentence boundaries. Semantic splitting follows meaning boundaries. Parent-child gives precise search and rich context. Most production systems use recursive splitting for speed and parent-child for quality.

#### Chunk Size Guidelines

```
128–256 tokens: precise embeddings, poor standalone answers
256–512 tokens: balanced, good for Q&A
512–1024 tokens: rich context, less precise embeddings
1024+ tokens: use as parent chunks only, not for direct retrieval
```

Overlap recommendation: 10–20% of chunk size. For 512-token chunks, 50–100 token overlap.

---

### Engineering Layer

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Generator


@dataclass
class Chunk:
    text: str
    token_count: int
    chunk_index: int
    parent_id: str | None = None   # for parent-child strategy


def _count_tokens(text: str) -> int:
    """Approximate token count — 4 characters ≈ 1 token for English."""
    return max(1, len(text) // 4)


def fixed_size_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Split by approximate token count with overlap."""
    words = text.split()
    words_per_chunk = chunk_size * 4 // 5  # ~5 chars per word on average
    overlap_words = overlap * 4 // 5
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(words):
        window = words[i : i + words_per_chunk]
        chunk_text = " ".join(window)
        chunks.append(Chunk(text=chunk_text, token_count=_count_tokens(chunk_text), chunk_index=idx))
        i += words_per_chunk - overlap_words
        idx += 1
    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Split on the most coarse separator that produces chunks near target size."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        parts = text.split(sep)
        result: list[str] = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if _count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if _count_tokens(part) > chunk_size:
                    result.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result

    raw_chunks = _split(text, separators)
    chunks: list[Chunk] = []
    for i, raw in enumerate(raw_chunks):
        # Add overlap by prepending tail of previous chunk
        if overlap > 0 and i > 0:
            prev_words = chunks[-1].text.split()
            overlap_text = " ".join(prev_words[-overlap:])
            raw = overlap_text + " " + raw
        chunks.append(Chunk(text=raw.strip(), token_count=_count_tokens(raw), chunk_index=i))
    return chunks


def parent_child_chunk(
    text: str,
    parent_size: int = 1024,
    child_size: int = 256,
    parent_id_prefix: str = "doc",
) -> tuple[list[Chunk], list[Chunk]]:
    """Return (parent_chunks, child_chunks). Child chunks reference their parent."""
    parents = recursive_chunk(text, chunk_size=parent_size, overlap=0)
    all_children: list[Chunk] = []
    for parent in parents:
        parent_id = f"{parent_id_prefix}_{parent.chunk_index}"
        children = recursive_chunk(parent.text, chunk_size=child_size, overlap=child_size // 4)
        for child in children:
            child.parent_id = parent_id
        all_children.extend(children)
    return parents, all_children
```

The `recursive_chunk` implementation tries the coarsest separator first (double newline = paragraph break) before falling back to finer separators (newline, sentence, word). This preserves paragraph structure when possible — the most important property for RAG quality. The overlap is added by prepending the tail of the previous chunk's words, not by re-splitting, which is cleaner than managing byte offsets.

---

### Architecture Layer

#### Chunking Strategy Selection

```
What kind of documents do you have?
      │
      ├── Homogeneous prose (policies, reports)?
      │         └── Recursive character splitting
      │               chunk_size=512, overlap=64
      │
      ├── Mixed content (docs with sections, code, tables)?
      │         └── Parent-child chunking
      │               parent=1024, child=256
      │
      ├── High-value documents (contracts, medical)?
      │         └── Semantic chunking
      │               (expensive, worth it for quality)
      │
      └── Unknown / mixed corpus?
                └── Recursive as default
                    A/B test with parent-child after launch
```

| Strategy | Retrieval quality | Ingestion cost | Implementation complexity | Best for |
|----------|-----------------|---------------|--------------------------|---------|
| Fixed-size | Low | Low | Minimal | Prototyping |
| Recursive | Medium | Low | Low | Most cases |
| Semantic | High | High (embed every sentence) | Medium | High-value docs |
| Parent-child | High | Medium | Medium | Q&A systems |

> **Bridge:** Your chunks are created. Each chunk needs labels — where it came from, who can see it, when it was last updated — so the retrieval system can filter results before returning them. That is the metadata design problem.

---

⚡ **Senior Checklist — 3.7**
- [ ] Never split mid-sentence — use recursive splitting before fixed-size splitting
- [ ] Set overlap to 10–20% of chunk size — a 512-token chunk needs 50–100 tokens of overlap
- [ ] Test chunk size with your actual LLM's context window: `(context_window - system_prompt - query) / k_chunks = max_chunk_size`
- [ ] For parent-child: embed child chunks, retrieve child IDs, return parent text to the LLM — never return raw child chunks
- [ ] Evaluate chunking strategy with RAGAS `context_recall`: if it is below 0.7, your chunks are too small or too large
- [ ] Keep code blocks intact — split around `\`\`\`` blocks, never through them
- [ ] Store both `chunk_index` and `total_chunks` as metadata — the model can use position context when assembling answers

---

## 3.8 Metadata Design

### 🧠 Mental Model
> Metadata is how you filter before you retrieve. A vector similarity search finds the most semantically similar chunks. Metadata filters ensure the model only sees chunks it is *allowed* to see, that are *fresh enough* to be trusted, and that belong to the *right context*. Without metadata, you retrieve the most similar text in the world — not the most relevant text in your tenant's allowed document set.

**Connects to:** Chunking (3.7) → **Metadata Design** → Retrieval Patterns (3.9)
**Parent concept:** Information retrieval architecture
**Builds on:** Section 3.7 — chunks are your retrieval units; metadata is the filter layer on top of similarity search

---

### Concept Layer

#### What Metadata Is For

You have a 1-million-vector index serving 200 enterprise tenants. A query from Tenant A should only return chunks from Tenant A's documents. A query about pricing should not return a chunk from a draft document that has not been approved. A query about a product should not return information about the discontinued version from 3 years ago.

Similarity search alone cannot enforce any of these constraints. A query for "pricing policy" will return the most similar chunk regardless of tenant, approval status, or age. Metadata filters are applied *before* or *during* similarity search to restrict the candidate set.

**The rule:** Design your metadata schema before you design your chunk schema. The metadata determines what filters you can apply; the filters determine whether your retrieval system is safe to use in production.

#### The Core Metadata Fields

Every chunk in a production RAG system should carry these fields:

```python
@dataclass
class ChunkMetadata:
    # Identity
    chunk_id: str              # unique ID for this chunk
    document_id: str           # parent document ID
    chunk_index: int           # position in document
    total_chunks: int          # total chunks in document

    # Access control
    tenant_id: str             # which organization owns this
    owner_user_id: str | None  # specific user owner (for personal docs)
    access_level: str          # "public", "internal", "confidential", "restricted"
    allowed_groups: list[str]  # ["finance", "legal"] — group-based access

    # Source traceability (for citations)
    source_path: str           # original file path or URL
    source_title: str          # human-readable document title
    page_number: int | None    # for PDFs
    section_heading: str | None  # for structured docs

    # Freshness
    created_at: str            # ISO 8601 timestamp
    updated_at: str            # ISO 8601 timestamp
    expires_at: str | None     # if None, never expires
    version: str               # document version string

    # Quality and classification
    content_type: str          # "policy", "manual", "faq", "contract", etc.
    language: str              # ISO 639-1 language code
    is_draft: bool             # draft documents excluded from standard retrieval
    confidence_score: float    # OCR quality score (1.0 = perfect, <0.7 = suspect)
```

#### Freshness Filtering

Stale documents are one of the top RAG failure modes. A chunk from a 2021 pricing document that has since been updated will produce confident wrong answers.

Two patterns for freshness:

**Pattern 1: Hard expiry.** Set `expires_at` when ingesting. Filter out expired chunks at query time. Simple but requires knowing expiry at index time.

**Pattern 2: Recency weighting.** Include `updated_at` in the metadata. At query time, apply a recency multiplier that decays similarity scores for older documents:

```
effective_score = similarity_score × recency_weight(updated_at)
recency_weight = 1.0 for docs updated in last 30 days
              = 0.9 for 30–90 days
              = 0.7 for 90–365 days
              = 0.5 for > 1 year
```

**In short:** Metadata is not optional labeling — it is the access control and freshness system for your retrieval layer. Design it as carefully as you design your database schema.

---

### Engineering Layer

```python
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    tenant_id: str
    access_level: str             # "public" | "internal" | "confidential" | "restricted"
    allowed_groups: list[str]
    source_path: str
    source_title: str
    created_at: str
    updated_at: str
    content_type: str
    language: str
    is_draft: bool = False
    expires_at: str | None = None
    page_number: int | None = None
    section_heading: str | None = None
    version: str = "1.0"
    confidence_score: float = 1.0

    @classmethod
    def create(
        cls,
        document_id: str,
        chunk_index: int,
        total_chunks: int,
        tenant_id: str,
        source_path: str,
        source_title: str,
        access_level: str = "internal",
        allowed_groups: list[str] | None = None,
        content_type: str = "document",
        language: str = "en",
        is_draft: bool = False,
        ttl_days: int | None = None,
        page_number: int | None = None,
        section_heading: str | None = None,
    ) -> "ChunkMetadata":
        now = datetime.now(timezone.utc).isoformat()
        expires_at = None
        if ttl_days is not None:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
        return cls(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            tenant_id=tenant_id,
            access_level=access_level,
            allowed_groups=allowed_groups or [],
            source_path=source_path,
            source_title=source_title,
            created_at=now,
            updated_at=now,
            content_type=content_type,
            language=language,
            is_draft=is_draft,
            expires_at=expires_at,
            page_number=page_number,
            section_heading=section_heading,
        )

    def is_accessible(self, user_tenant_id: str, user_groups: list[str]) -> bool:
        """Enforce access control before retrieval."""
        if self.tenant_id != user_tenant_id:
            return False
        if self.is_draft:
            return False
        if self.expires_at:
            if datetime.fromisoformat(self.expires_at) < datetime.now(timezone.utc):
                return False
        if self.access_level == "public":
            return True
        if self.access_level == "internal":
            return True  # any authenticated user in this tenant
        if self.access_level in ("confidential", "restricted"):
            return bool(set(self.allowed_groups) & set(user_groups))
        return False

    def recency_weight(self) -> float:
        updated = datetime.fromisoformat(self.updated_at)
        age_days = (datetime.now(timezone.utc) - updated).days
        if age_days <= 30:
            return 1.0
        elif age_days <= 90:
            return 0.9
        elif age_days <= 365:
            return 0.7
        return 0.5

    def to_payload(self) -> dict[str, Any]:
        """Serialize for storage in vector DB payload."""
        return asdict(self)
```

The `is_accessible` method enforces three access checks in sequence: tenant isolation first (hard boundary), draft exclusion second (prevents incomplete content), then expiry, then group-based access control. This order matters — checking tenant first ensures tenant isolation cannot be bypassed by any other condition.

---

### Architecture Layer

#### Metadata Filter Architecture

```
[Query] + [User context: tenant_id, groups]
      │
      ↓
[Metadata Filter Builder]
      ├── tenant_id = user.tenant_id          (hard filter, always)
      ├── is_draft = false                    (hard filter, always)
      ├── expires_at > now (or null)          (hard filter, always)
      └── access_level in user.allowed_levels (hard filter, always)
      │
      ↓ filter spec
[Vector DB Search]
      ├── Qdrant: payload filter applied at index scan time
      └── pgvector: WHERE clause in SQL
      │
      ↓ filtered candidates
[Recency weighting] (optional, apply score × recency_weight)
      │
      ↓ final ranked results
[Return to retrieval layer]
```

| Filter type | Applied at | Performance impact | Notes |
|------------|-----------|-------------------|-------|
| Tenant isolation | Index scan | ~0ms (Qdrant HNSW payload filter) | Must be first |
| Draft exclusion | Index scan | ~0ms | Prevents stale content |
| Expiry check | Index scan | ~0ms | Requires indexed field |
| Group access | Index scan | ~0ms | Use array contains |
| Recency weighting | Post-retrieval | ~1ms per result | Score adjustment, not filter |

> **Bridge:** Your chunks have metadata and your index is ready. Now you need to query it — and the difference between top-k retrieval, hybrid search, and reranking determines whether your system finds the right documents or just the most similar ones.

---

⚡ **Senior Checklist — 3.8**
- [ ] Always filter by `tenant_id` first — tenant isolation must be enforced at the index scan level, not in application code after retrieval
- [ ] Store `expires_at` as an indexed field in your vector DB — expiry checks must happen at scan time, not post-retrieval
- [ ] Include `section_heading` and `page_number` in metadata — the model needs them to construct accurate citations
- [ ] Set `is_draft=True` for documents pending approval and exclude them from standard retrieval filters
- [ ] Add `content_type` to every chunk — this allows content-type-specific retrieval (only search FAQs for FAQ-style queries)
- [ ] Version your metadata schema with a `schema_version` field — future schema changes require migration, not corruption
- [ ] Log metadata filter statistics: how often each filter removes results — high removal rates indicate access control issues

---

## 3.9 Retrieval Patterns

### 🧠 Mental Model
> Dense retrieval finds what *means* the same thing. BM25 finds what *says* the same thing. Reranking finds what *best answers* the specific question. Each layer fixes what the previous layer misses. A production RAG system needs at least the first two, and the third for high-stakes queries.

**Connects to:** Metadata Design (3.8) → **Retrieval Patterns** → Prompt Assembly (3.10)
**Parent concept:** Information retrieval strategies
**Builds on:** Section 3.4 — vector search finds semantically similar chunks; this section extends it with keyword search and reranking

---

### Concept Layer

#### Pattern 1: Top-k Dense Retrieval

The simplest pattern: embed the query, search the vector index, return the k most similar chunks.

```
Query → embed → search HNSW → top-k chunks
```

**When it works:** When the query and the document use similar vocabulary or when semantic paraphrase matters (user asks "how to cancel" and document says "steps to terminate subscription").

**When it fails:** When the query contains specific technical terms, product names, model numbers, or error codes. A query for "error code E-4092" will not match a document about "E-4092 calibration failure" through dense retrieval if those exact tokens are not well-represented in the embedding training data.

**k values:**
- 3–5: for chat interfaces where context window is constrained
- 10–20: for research or comprehensive answer modes
- 50+: for reranking pipelines where you over-retrieve and then filter

#### Pattern 2: BM25 (Keyword) Retrieval

BM25 (Best Match 25) is the standard keyword ranking algorithm used by Elasticsearch and Solr. It scores documents by term frequency, inverse document frequency, and document length normalization.

```
score(doc, query) = Σ IDF(term) × (TF(term, doc) × (k1 + 1)) / (TF(term, doc) + k1 × (1 - b + b × |doc| / avgdl))

Where:
  IDF = log((N - df + 0.5) / (df + 0.5)) — how rare is this term across the corpus
  TF = raw term frequency in the document
  k1 = 1.5 (term saturation — diminishing returns for repeated terms)
  b = 0.75 (length normalization)
  avgdl = average document length in corpus
```

**When BM25 wins over dense:**
- Exact product names: "iPhone 15 Pro Max" — BM25 matches exactly; dense may match "iPhone 14" if semantically similar
- Error codes: "NullPointerException at line 482" — BM25 matches the specific code; dense matches generic "error" content
- Model numbers, part numbers, CVE IDs, SKUs — any identifier where exact match matters

**In short:** Dense retrieval understands meaning. BM25 finds exact terms. Use both and combine their scores.

#### Pattern 3: Hybrid Search (Dense + BM25)

Combine both scores using Reciprocal Rank Fusion (RRF) — a rank-based fusion method that is robust to score scale differences.

```python
RRF_score(doc) = Σ 1 / (k + rank_in_list)

Where k=60 (constant that smooths the rank curve)
      rank_in_list = position in the ranked list (1-indexed)
```

For each document, sum `1 / (60 + rank)` across all retrieval lists it appears in. This works **because** it does not require normalizing scores from different systems to the same scale — which means **in practice** you can combine BM25 (scores of 0–25) with cosine similarity (scores of 0–1) without any calibration.

**Typical improvement from hybrid over dense alone:** 5–15% better NDCG@10 on most benchmarks. The improvement is larger for corpora with technical terminology.

#### Pattern 4: Reranking

After retrieving a candidate set (20–50 chunks), pass every (query, chunk) pair through a **cross-encoder reranker** — a model that takes query AND chunk as input together and produces a single relevance score.

Dense embeddings: encode query and document *separately* → fast but misses fine-grained relevance
Cross-encoder: encode query and document *together* → slower but sees exactly how they relate

```
[Query + Chunk 1] → cross-encoder → score 0.92
[Query + Chunk 2] → cross-encoder → score 0.43
[Query + Chunk 3] → cross-encoder → score 0.88
→ Reranked: Chunk 1, Chunk 3, Chunk 2
```

**Latency cost:** 50–200ms extra per query for a cross-encoder on 20 candidates.
**Quality gain:** 10–25% better NDCG@10 vs dense-only retrieval.
**When to use:** High-stakes queries where answer quality matters more than latency. Not justified for every query in a high-volume system.

Models to use:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (6-layer, fast, 5ms/pair)
- `cross-encoder/ms-marco-electra-base` (more accurate, 20ms/pair)
- `Cohere rerank-english-v3.0` (cloud API, 50–100ms for 20 docs)

**In short:** Dense retrieval is fast but misses exact terms. BM25 finds exact terms but misses meaning. Hybrid covers both. Reranking re-scores the top candidates with full cross-attention. Use all three when quality is critical.

---

### Engineering Layer

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int
    source: str   # "dense" | "bm25" | "hybrid" | "reranked"
    metadata: dict[str, Any]


class HybridRetriever:
    """Combines dense vector search + BM25 using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store,         # any object with .search(embedding, tenant_id, k) method
        embedder,             # any object with .embed(text) -> EmbeddingResult
        corpus_chunks: list[tuple[str, str, dict]],  # (chunk_id, text, metadata)
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.rrf_k = rrf_k

        # Build BM25 index
        self._chunk_ids = [c[0] for c in corpus_chunks]
        self._chunk_texts = {c[0]: c[1] for c in corpus_chunks}
        self._chunk_meta = {c[0]: c[2] for c in corpus_chunks}
        tokenized = [c[1].lower().split() for c in corpus_chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        k: int = 5,
        dense_k: int = 20,
        bm25_k: int = 20,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        # Dense retrieval
        embedding = self.embedder.embed(query).vector
        dense_results = self.vector_store.search(embedding, tenant_id, k=dense_k)
        dense_rank = {r.id: i + 1 for i, r in enumerate(dense_results)}

        # BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_k]
        bm25_rank = {self._chunk_ids[i]: rank + 1 for rank, i in enumerate(bm25_top_indices)}

        # Reciprocal Rank Fusion
        all_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        rrf_scores: dict[str, float] = {}
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[chunk_id])
            if chunk_id in bm25_rank:
                score += 1.0 / (self.rrf_k + bm25_rank[chunk_id])
            rrf_scores[chunk_id] = score

        # Build final ranked list
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results: list[RetrievalResult] = []
        for rank, (chunk_id, score) in enumerate(ranked):
            if score < min_score:
                break
            text = self._chunk_texts.get(chunk_id, "")
            meta = self._chunk_meta.get(chunk_id, {})
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                rank=rank + 1,
                source="hybrid",
                metadata=meta,
            ))
        return results


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
) -> list[RetrievalResult]:
    """Rerank candidates using a cross-encoder."""
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder(model_name)
    pairs = [(query, r.text) for r in candidates]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    output: list[RetrievalResult] = []
    for rank, (result, score) in enumerate(reranked[:top_n]):
        output.append(RetrievalResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=float(score),
            rank=rank + 1,
            source="reranked",
            metadata=result.metadata,
        ))
    return output
```

The RRF implementation uses a dictionary union across dense and BM25 result sets — a document appearing in both lists gets contributions from both rank positions, which is why hybrid consistently outperforms either alone. The `source` field on each result tracks provenance, which is essential for debugging retrieval failures: if you see only `bm25` results, your dense index may be broken.

---

### Architecture Layer

#### Retrieval Pipeline with Quality Tiers

```
[User Query]
      │
      ├─────────────────────────────────────────┐
      │                                         │
      ↓ embed query (100ms)                     ↓ tokenize
[Dense retrieval]                          [BM25 retrieval]
  top-20 by cosine                         top-20 by BM25 score
      │                                         │
      └──────────────┬──────────────────────────┘
                     ↓
              [RRF Fusion]
               top-10 candidates
                     │
            ┌────────┴─────────┐
            │                  │
      [< 100ms SLA]       [> 100ms OK]
            │                  │
         return top-5    [Cross-encoder rerank]
                          top-5 from 10 candidates
                               │
                          return reranked top-5
```

| Strategy | Latency added | Recall improvement | Use when |
|----------|-------------|-------------------|---------|
| Dense only | 0ms baseline | Baseline | Prototyping |
| +BM25 hybrid | +5ms | +5–15% | All production systems |
| +Cross-encoder rerank | +50–200ms | +10–25% | High-stakes queries, SLA allows |

> **Bridge:** You now have a ranked list of relevant chunks. The next problem is packing them into a prompt that the model can actually use — without overflowing the context window or burying the most relevant chunk at the bottom where the model ignores it.

---

⚡ **Senior Checklist — 3.9**
- [ ] Always run hybrid retrieval (dense + BM25) in production — dense-only misses exact technical terms and identifiers
- [ ] Use RRF for score fusion, not score normalization — RRF is robust to score scale differences between systems
- [ ] Over-retrieve for reranking: fetch 20–50 candidates, rerank, return top-5 — never rerank only 5 candidates
- [ ] Track `source` field per result in your retrieval logs — "all results are BM25" indicates a broken dense index
- [ ] Set `rrf_k=60` as your default RRF constant — lower values over-weight the top-ranked results, higher values flatten the curve
- [ ] Add minimum score threshold (0.70 for cosine) to filter low-confidence results even if fewer than k results remain
- [ ] Benchmark reranker latency on your p95 query load — cross-encoders add 50–200ms; verify your SLA can absorb this

---

## 3.10 Prompt Assembly and Context Packing

### 🧠 Mental Model
> Imagine you are a chef who can only plate five ingredients. You retrieved twenty from the pantry -- now decide which five, in what order, and how much of each. Prompt assembly is exactly that: given a token budget and a ranked list of retrieved chunks, decide what goes in, in what order, and how much space each gets.

**Connects to:** Retrieval Patterns (3.9) -> **Prompt Assembly** -> Citation and Grounding (3.11)
**Parent concept:** Context window management
**Builds on:** Section 3.9 -- retrieval gives you a ranked list of chunks; assembly decides how to fit them into the prompt without exceeding the model's context limit

---

### Concept Layer

#### The Token Budget Problem

Think of a model's context window like a notebook with a fixed number of pages. You cannot add more pages. Every piece of text you insert -- your instructions, the user question, the retrieved documents, and space for the model's answer -- all compete for those same pages.

Claude Sonnet 4.6 has a 200,000-token context window. That sounds huge, but it fills up quickly:

```
200,000 tokens total window
  -  2,000  system prompt (instructions, persona, rules)
  -    500  user query
  -  4,000  reserved for model answer
  -    200  formatting overhead (XML tags, separators)
  --------------------------------------------------
  193,300  tokens available for retrieved chunks
  /    512  tokens per chunk (average)
  --------------------------------------------------
  ~377     maximum chunks you can fit
```

For smaller models the math is tighter:

```
GPT-4o (128K window):     ~104K tokens for chunks, ~203 chunks at 512 tokens
Llama 3 8B (8K window):   ~5.5K tokens for chunks, ~10 chunks at 512 tokens
```

**The rule:** Always calculate your token budget before assembly: `budget = window_size - system_tokens - query_tokens - answer_reserve`. Apply 15% headroom. Never fill the window completely.

#### The Lost-in-the-Middle Problem

Here is a counterintuitive research finding: language models do not pay equal attention to all parts of their context. When you give a model ten chunks of text, it pays the most attention to the **first few** and the **last few**. Chunks buried in the middle receive less attention -- even when they contain the most relevant information.

Think of a long conference where five speakers present back to back. You remember the first speaker (you were fresh) and the last speaker (most recent). The third speaker is a blur.

Language models behave this way. Your retrieval system ranks chunks by relevance score. If you naively pack from most-relevant to least-relevant, your second-best chunk lands in the forgettable middle.

**The solution -- sandwich layout:** Put the most relevant chunk first, the second-most-relevant last, and fill the middle with the rest.

```
Sandwich layout:
  Position 1:     Most relevant        <- high attention
  Position 2..N-1: Supporting chunks   <- lower attention
  Position N:     Second-most relevant <- high attention
```

**In short:** The model remembers beginnings and endings. Put your two best chunks at those positions.

#### Context Ordering Strategies

**Strategy 1: Relevance-first (naive).** Pack from highest to lowest score. Simple, but second-best chunk is forgotten in the middle.

**Strategy 2: Sandwich layout.** Best first, second-best last. Free quality improvement -- 5-15% better answers.

**Strategy 3: Document-order grouping.** Group chunks from the same source in their original page order. Better for sequence questions: "what are the steps in this process?"

**Strategy 4: MMR diversity injection.** Your top-5 most similar chunks may all say the same thing in slightly different words -- wasting context on redundancy. MMR (Maximal Marginal Relevance) picks chunks that are both relevant AND different from each other. Like picking the five most informative briefing documents, not just the five most similar ones.

**The rule:** Default to sandwich. Use document-order for process questions. Use MMR when top-k results are paraphrases of each other.

#### Token Counting Is Safety-Critical

If your prompt exceeds the context window, the model silently truncates the end. No error, no warning. The model answers based on whatever fit. If your most relevant chunk was last, it vanishes completely.

Every production RAG system counts tokens before assembly, using `tiktoken` for OpenAI models or the Anthropic SDK token counting endpoint for Claude.

**In short:** Count tokens before assembly. Truncation is silent. A wrong answer from invisible truncation is the hardest bug to diagnose.

---

### Engineering Layer

```python
from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass
class ContextChunk:
    chunk_id: str
    text: str
    score: float
    source_title: str
    page_number: int | None
    section_heading: str | None


@dataclass
class AssembledContext:
    chunks: list[ContextChunk]
    total_tokens: int
    budget_used_pct: float
    truncated: bool


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except KeyError:
        return len(text) // 4  # Claude and other non-tiktoken models


def assemble_context(
    chunks: list[ContextChunk],
    system_prompt: str,
    user_query: str,
    model: str = "gpt-4o",
    context_window: int = 128_000,
    answer_reserve: int = 4_000,
    layout: str = "sandwich",
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
) -> AssembledContext:
    overhead_tokens = (
        count_tokens(system_prompt, model)
        + count_tokens(user_query, model)
        + answer_reserve
        + 200
    )
    budget = int(context_window * 0.85) - overhead_tokens

    if use_mmr and len(chunks) > 1:
        chunks = _mmr_select(chunks, mmr_lambda=mmr_lambda)

    packed: list[ContextChunk] = []
    total = 0
    for chunk in chunks:
        tokens = count_tokens(chunk.text, model)
        if total + tokens > budget:
            break
        packed.append(chunk)
        total += tokens

    if layout == "sandwich" and len(packed) > 2:
        first = packed[0]
        second_best = packed[1]
        middle = packed[2:]
        packed = [first] + middle + [second_best]

    return AssembledContext(
        chunks=packed,
        total_tokens=total,
        budget_used_pct=round(total / max(budget, 1) * 100, 1),
        truncated=len(packed) < len(chunks),
    )


def _mmr_select(chunks: list[ContextChunk], mmr_lambda: float = 0.7) -> list[ContextChunk]:
    if not chunks:
        return []
    selected: list[ContextChunk] = [chunks[0]]
    remaining = list(chunks[1:])
    while remaining:
        best_chunk = None
        best_score = float("-inf")
        for candidate in remaining:
            redundancy = max(_text_overlap(candidate.text, s.text) for s in selected)
            score = mmr_lambda * candidate.score - (1 - mmr_lambda) * redundancy
            if score > best_score:
                best_score = score
                best_chunk = candidate
        if best_chunk is None:
            break
        selected.append(best_chunk)
        remaining = [c for c in remaining if c.chunk_id != best_chunk.chunk_id]
    return selected


def _text_overlap(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def format_context_block(assembled: AssembledContext) -> str:
    lines: list[str] = ["<context>"]
    for i, chunk in enumerate(assembled.chunks, start=1):
        source = chunk.source_title
        if chunk.page_number:
            source += f", p.{chunk.page_number}"
        if chunk.section_heading:
            source += f" -- {chunk.section_heading}"
        lines.append(f"[{i}] Source: {source}")
        lines.append(chunk.text)
        lines.append("")
    lines.append("</context>")
    return "\n".join(lines)
```

The `assemble_context` function enforces token budget, optional MMR diversity, and layout optimization in a single pass. The sandwich reorder swaps the second item to the last position -- that single change improves answer quality with zero compute cost. The `_mmr_select` function penalizes candidates that overlap heavily with already-selected chunks, preventing the context from becoming repetitive paraphrases.

---

### Architecture Layer

#### Context Assembly Pipeline

```
[Ranked chunks from retrieval]
         |
         v
[Token Budget Calculator]
  budget = window * 0.85 - overhead_tokens
         |
         v
[MMR Diversity Filter] (optional)
  removes chunks with Jaccard overlap > 0.5
         |
         v
[Budget Packer]
  adds chunks until budget exhausted
  truncated=True if any chunks dropped
         |
         v
[Layout Reorderer]
  sandwich: [best, middle..., second-best]
  document: group by source, sort by chunk_index
         |
         v
[Context Block Formatter]
  <context>
  [1] Source: Policy.pdf, p.3 -- Refund Section
  ...text...
  [2] Source: FAQ.pdf -- Returns
  ...text...
  </context>
         |
         v
[Final prompt -> LLM]
```

#### Context Window Comparison

| Model | Context window | Safe chunk budget | Chunks at 512 tokens |
|-------|--------------|------------------|---------------------|
| GPT-4o | 128K | ~104K | ~203 |
| Claude Sonnet 4.6 | 200K | ~163K | ~318 |
| Claude Haiku 4.5 | 200K | ~163K | ~318 |
| Gemini 1.5 Pro | 1M | ~816K | ~1,593 |
| Llama 3 8B | 8K | ~5.5K | ~10 |

> **Bridge:** Your chunks are packed into the prompt with numbered references. Now the model needs to answer using those references and tell the user which chunk each fact came from. That is the citation and grounding problem.

---

⚡ **Senior Checklist -- 3.10**
- [ ] Calculate token budget before assembly: `window * 0.85 - overhead` -- the 15% headroom prevents silent truncation on edge-case long inputs
- [ ] Use sandwich layout (best chunk first AND last) to counter lost-in-the-middle -- zero extra cost, measurable quality gain
- [ ] Log `truncated=True` events and alert when rate exceeds 5% of queries -- indicates chunk size or k is too large
- [ ] Run MMR when top-k results have Jaccard overlap above 0.5 -- context full of paraphrases wastes token budget
- [ ] Wrap context in `<context>...</context>` XML tags -- Claude and GPT-4 hallucinate measurably less from tagged structured input
- [ ] Number every chunk [1], [2] in the context block -- these numbers are the citation anchors for the next layer
- [ ] Test with your longest real system prompts -- prompts grow over time as product rules are added, silently shrinking chunk budget


## 3.11 Citation and Grounding Patterns

### 🧠 Mental Model
> A citation is a debugging tool disguised as a politeness feature. When the model says "the refund window is 30 days [3]", you open chunk 3, find the source document, and verify the fact in 30 seconds. Without citations, a wrong answer has no trail. With citations, every wrong answer is a solvable problem.

**Connects to:** Prompt Assembly (3.10) -> **Citations** -> Access Control (3.12)
**Parent concept:** Answer grounding and auditability
**Builds on:** Section 3.10 -- numbered chunks in the context block are the raw material; this section shows how to instruct the model to cite them accurately

---

### Concept Layer

#### Why Citations Are Non-Negotiable in Production

Imagine you run a customer support system for a bank. The model answers: "wire transfers over $10,000 require a 24-hour hold." Is that correct? If you cannot trace that answer to a specific policy document with a specific version date, you cannot verify it, cannot update it when the policy changes, and cannot defend it to a regulator.

Think of citations like references in a school paper. A claim without a citation is just an opinion. A claim with a citation is a verifiable fact. In a production RAG system, the difference between the two is the difference between a trustworthy product and a liability.

Citations serve three roles:

**1. Trust signal for users.** "According to your Employee Handbook, p.12" feels different from "I think." Users who see a cited source are measurably more likely to trust and act on the answer.

**2. Debugging trail for engineers.** When the model gives a wrong answer, the citation points you to the exact chunk to fix, exclude, or update. Without a citation, you are searching blindly. With one, fixing the answer is a one-minute task.

**3. Compliance evidence.** Healthcare, finance, and legal systems must show an audit trail from question to answer to source. RAG with verified citations is the only architecture that provides this trace.

**The rule:** Build citation infrastructure on day one. Retrofitting it into a system never designed for citations requires rewriting the entire prompt layer.

#### The Citation Injection Pattern

The mechanism is simple: number each chunk in the context block (done in section 3.10), then instruct the model to reference those numbers inline.

```
System prompt:
  "Answer using ONLY the information in <context>.
   After every factual statement, add [N] where N is the chunk number.
   If the answer is not in the context, say:
   'I do not have that information in the provided documents.'
   Never use knowledge from outside the context."

Model output:
  "Returns are accepted within 30 days [1].
   Items over $100 require manager approval before a refund is issued [2]."
```

This works because language models were pre-trained on billions of documents that use citation patterns -- academic papers, legal texts, Wikipedia footnotes. The model already knows how to cite; the instruction activates behavior it already learned.

#### Grounding Verification -- Did the Model Read Accurately?

A citation tells you which chunk the model referenced. It does not guarantee the model read it accurately. A model might write "the refund window is 60 days [1]" when chunk 1 says "30 days." The citation is present but the fact is wrong.

**Grounding verification** asks: does the model's answer actually reflect what the cited chunks say? You are checking whether someone reading only the cited chunks could arrive at the same answer.

A practical proxy: for every factual claim in the answer, check whether the key content words appear in the cited chunk. "60 days" in the answer but no mention of "60" in the cited chunk -- that is a grounding failure.

**In short:** Citations point to the source. Grounding verification confirms the model read that source accurately. Both are required for a trustworthy answer pipeline.

#### Prompt Patterns That Reduce Hallucination

**Pattern 1: Source constraint (always use this)**
Explicitly forbid the model from using knowledge outside the context. This alone reduces factual hallucination by 40-60% compared to baseline RAG without constraints.

**Pattern 2: Uncertainty disclosure**
Tell the model to use hedged language when the context is thin: "According to [1]..." surfaces uncertainty instead of hiding it.

**Pattern 3: Structured output format (use when answer feeds automation)**
Force a specific output structure so citations are programmatically parseable. Free-text citation parsing is fragile.

```
ANSWER: [answer with inline [N] citations]
SOURCES: [comma-separated numbers, e.g. "1, 3"]
CONFIDENCE: [high / medium / low]
```

---

### Engineering Layer

```python
from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic


@dataclass
class CitedAnswer:
    answer_text: str
    cited_chunk_ids: list[str]     # chunk UUIDs mapped from [N] numbers
    uncited_sentences: list[str]   # sentences with no [N] citation
    confidence: str                # "high" | "medium" | "low"
    grounding_score: float         # 0.0-1.0: fraction of answer content in cited chunks


CITATION_SYSTEM_PROMPT = (
    "You are a knowledge assistant. Answer questions using ONLY the information "
    "in <context>.\n\n"
    "Rules:\n"
    "1. After every factual statement, add [N] where N is the context chunk number.\n"
    "2. If the answer is not in the context: respond \'I do not have that "
    "information in the provided documents.\'\n"
    "3. Never use knowledge from outside the context.\n"
    "4. Respond in this exact format:\n"
    "   ANSWER: [your answer with inline [N] citations]\n"
    "   SOURCES: [comma-separated citation numbers, e.g. \'1, 3\']\n"
    "   CONFIDENCE: [high if context clearly answers / medium if partial / low if thin]"
)


def parse_cited_answer(
    raw_response: str,
    chunk_id_map: dict[int, str],   # {1: "uuid-abc", 2: "uuid-xyz"}
    context_chunks: dict[int, str], # {1: "chunk text", 2: "chunk text"}
) -> CitedAnswer:
    answer_match = re.search(r"ANSWER:\s*(.+?)(?=SOURCES:|CONFIDENCE:|$)", raw_response, re.DOTALL)
    sources_match = re.search(r"SOURCES:\s*(.+?)(?=CONFIDENCE:|$)", raw_response, re.DOTALL)
    confidence_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", raw_response, re.IGNORECASE)

    answer_text = answer_match.group(1).strip() if answer_match else raw_response
    sources_raw = sources_match.group(1).strip() if sources_match else ""
    confidence = confidence_match.group(1).lower() if confidence_match else "low"

    cited_numbers = [int(n) for n in re.findall(r"\d+", sources_raw)]
    cited_chunk_ids = [chunk_id_map[n] for n in cited_numbers if n in chunk_id_map]

    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    uncited = [s for s in sentences if s.strip() and not re.search(r"\[\d+\]", s)]

    grounding_score = _compute_grounding_score(answer_text, cited_numbers, context_chunks)

    return CitedAnswer(
        answer_text=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        uncited_sentences=uncited,
        confidence=confidence,
        grounding_score=grounding_score,
    )


def _compute_grounding_score(
    answer: str,
    cited_numbers: list[int],
    context_chunks: dict[int, str],
) -> float:
    if not cited_numbers:
        return 0.0
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and", "or"}
    answer_words = set(answer.lower().split()) - stopwords
    if not answer_words:
        return 1.0
    cited_text = " ".join(context_chunks.get(n, "") for n in cited_numbers)
    cited_words = set(cited_text.lower().split()) - stopwords
    return len(answer_words & cited_words) / len(answer_words)


def get_rag_answer(
    client: anthropic.Anthropic,
    context_block: str,
    user_query: str,
    chunk_id_map: dict[int, str],
    context_chunks: dict[int, str],
    model: str = "claude-sonnet-4-6",
) -> CitedAnswer:
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=CITATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{context_block}\n\nQuestion: {user_query}"}],
    )
    return parse_cited_answer(response.content[0].text, chunk_id_map, context_chunks)
```

The `_compute_grounding_score` removes stopwords before comparing answer words to cited chunk words. Without this filter, common words like "the" and "is" inflate scores regardless of factual accuracy -- every answer would score 0.9+ even when fabricated. A grounding score below 0.4 is a reliable hallucination flag: the model wrote things that simply do not appear in the chunks it cited.

---

### Architecture Layer

#### Citation and Grounding Pipeline

```
[LLM raw response]
  "ANSWER: Refund window is 30 days [1].
   SOURCES: 1
   CONFIDENCE: high"
         |
         v
[Citation Parser]
  extract ANSWER / SOURCES / CONFIDENCE
  map [1] -> chunk UUID "abc-123"
         |
         v
[Uncited Sentence Detector]
  flag sentences with no [N] -- model going off-context
         |
         v
[Grounding Verifier]
  answer words vs cited chunk words -> score 0.0-1.0
  score < 0.4 -> add "may be incomplete" warning to UI
         |
         v
[Audit Log]
  query_hash, chunk_ids, grounding_score, confidence, latency_ms
  (do NOT store raw query text if it may contain PII)
         |
         v
[Response to client: answer_text + cited_chunk_ids + grounding_score]
  UI renders [N] as a hover card showing source title, page, excerpt
```

| Citation approach | Parseability | User trust | Effort |
|------------------|-------------|-----------|--------|
| Structured ANSWER/SOURCES | High -- regex | Medium | Low |
| Inline [N] in prose | Medium | High (visible) | Low |
| UI hover cards showing source excerpt | N/A | Very high | Medium |
| No citations | None | Zero | Never use |

> **Bridge:** Your system cites sources and verifies grounding. But if 200 enterprise customers share one index, one customer must never see another customer's citations. That requires access control enforced at the database layer -- not in application code.

---

⚡ **Senior Checklist -- 3.11**
- [ ] Always use structured output (ANSWER/SOURCES/CONFIDENCE) -- parsing citations from free-form prose breaks on multi-paragraph answers
- [ ] Map citation numbers back to chunk UUIDs before returning to clients -- numbers are session-local and meaningless after the session ends
- [ ] Alert when 7-day average grounding_score falls below 0.5 -- trending downward means chunk quality is degrading
- [ ] Count uncited_sentences per response and alert when ratio exceeds 20% -- the model is consistently going off-context
- [ ] Show "I do not have that information" gracefully in the UI -- never let low-confidence hallucinations reach users unlabeled
- [ ] Test citation parsing against adversarial model outputs: "SOURCES: none", missing sections, letters instead of numbers
- [ ] Store raw LLM responses alongside parsed versions for 30 days -- needed to debug citation failures users report days later


## 3.12 Access Control and Tenant Isolation

### 🧠 Mental Model
> Think of a multi-tenant RAG system like a bank with safety deposit boxes. Each customer has their own box. The teller must check the key before opening any box. If you build the teller to work without checking keys -- even accidentally -- you have a security breach, not a bug. Tenant isolation is a security property, not a convenience feature.

**Connects to:** Citations (3.11) -> **Access Control** -> Index Freshness (3.13)
**Parent concept:** Multi-tenant security architecture
**Builds on:** Section 3.8 -- metadata schema designed the access control fields; this section shows how to enforce them at every layer of the stack

---

### Concept Layer

#### Why Post-Retrieval Filtering Is Not Enough

The naive approach: retrieve everything, then filter out documents the user cannot see in application code after retrieval.

```python
# DANGEROUS -- do not do this
results = vector_db.search(query_embedding, k=100)
allowed = [r for r in results if r.tenant_id == user.tenant_id]
return allowed[:5]
```

This fails in three ways -- and each failure mode is worse than the previous:

**Failure 1: Wrong results returned.** You retrieve 100 documents, filter to the 8 your user is allowed to see, then return the top 5 of those 8. But the real top-5 most relevant documents for this user might all have been filtered out. You are not returning the best results; you are returning the best results from a biased sample.

**Failure 2: Existence leakage.** An attacker who notices response latency varies based on how many results were filtered can infer that certain document categories exist in other tenants -- even though they never see the content.

**Failure 3: One bug away from a breach.** If the filter condition has a typo -- wrong variable name, wrong comparison operator -- all documents become visible to all users. A single incorrect line of application code collapses your entire isolation boundary.

**The rule:** Tenant isolation must be enforced at the index scan level, before similarity scoring, not in application code after retrieval.

#### Index-Level Filtering -- How It Works

Both Qdrant and pgvector support pre-filtering: the filter is applied to the index scan itself, so documents not matching the filter are never considered as candidates at all.

**Qdrant payload filtering:**
Qdrant stores metadata (called "payload") alongside every vector. When you search, you pass a filter condition. Qdrant integrates the payload filter into the HNSW graph traversal -- nodes that do not match the filter are skipped during graph navigation, not just excluded from final results. The non-matching documents are invisible to the search, not just removed afterwards.

Imagine searching a city by walking streets on a map. Payload filtering is like having certain streets literally removed from the map before you start walking. Post-retrieval filtering is like walking all the streets, then throwing away directions to certain destinations after you arrive. The first approach means you never even go near those destinations. The second means the information was already in your head.

**pgvector with Row-Level Security (RLS):**
PostgreSQL Row-Level Security allows you to define access policies at the database level. With RLS enabled, a query running as `tenant_acme` only ever sees rows where `tenant_id = acme-corp` -- even if the application code forgets to add a WHERE clause.

```sql
-- Enable RLS on the chunks table
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Policy: a session can only see chunks for its current tenant
CREATE POLICY tenant_isolation ON document_chunks
  USING (tenant_id = current_setting('app.current_tenant'));

-- Application sets this at connection time
SET LOCAL app.current_tenant = 'acme-corp';

-- This query automatically only sees acme-corp chunks
-- even with no WHERE clause in application code
SELECT * FROM document_chunks
ORDER BY embedding <=> $1 LIMIT 5;
```

RLS is defense-in-depth: even if your application code has a bug that forgets to filter by tenant, the database enforces isolation at the storage layer.

**In short:** Use index-level filtering so tenant data is invisible at the database layer. Post-retrieval application filtering is a single bug away from a catastrophic breach.

#### Namespace Isolation -- The Strongest Guarantee

Some vector databases (Pinecone, Weaviate) support namespaces -- completely separate logical partitions of the same index where vectors in namespace A are never accessible when querying namespace B.

Namespace isolation is structurally impossible to bypass: there is no filter to forget, no WHERE clause to omit. The data in one namespace simply does not exist in another namespace's search space.

The tradeoff: cross-namespace operations (like an admin search that spans all tenants) require querying each namespace separately.

| Isolation method | Security strength | Admin search support | Setup complexity |
|-----------------|-----------------|--------------------|-----------------|
| Payload filter | High | Easy (omit filter) | Low |
| Row-level security | Very high (DB enforces) | Requires superuser | Medium |
| Namespace isolation | Absolute (structural) | Manual loop | Medium |

**The rule:** Use payload filters for SaaS products with many tenants. Use namespaces for regulated workloads where a configuration error must be structurally impossible to cause a breach.

---

### Engineering Layer

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


@dataclass
class UserContext:
    user_id: str
    tenant_id: str          # MUST come from validated JWT, never from request body
    groups: list[str]       # ["finance", "legal"]
    access_levels: list[str]  # ["public", "internal"] -- derived from user role


def build_tenant_filter(ctx: UserContext) -> Filter:
    return Filter(
        must=[
            # Hard boundary 1: tenant isolation -- never relax
            FieldCondition(key="tenant_id", match=MatchValue(value=ctx.tenant_id)),
            # Hard boundary 2: exclude draft documents
            FieldCondition(key="is_draft", match=MatchValue(value=False)),
        ],
        should=[
            # At least one of the user's allowed access levels must match
            *[
                FieldCondition(key="access_level", match=MatchValue(value=level))
                for level in ctx.access_levels
            ]
        ],
        minimum_should_match=1,
    )


class TenantAwareVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection = collection_name

    def search(
        self,
        query_vector: list[float],
        user_context: UserContext,
        k: int = 10,
        score_threshold: float = 0.70,
    ) -> list[dict[str, Any]]:
        tenant_filter = build_tenant_filter(user_context)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=tenant_filter,
            limit=k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "chunk_id": r.payload.get("chunk_id"),
                "text": r.payload.get("text", ""),
                "score": r.score,
                "source_title": r.payload.get("source_title", ""),
                "tenant_id": r.payload.get("tenant_id"),
                "access_level": r.payload.get("access_level"),
            }
            for r in results
        ]

    def upsert_chunk(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        from qdrant_client.models import PointStruct
        if "tenant_id" not in payload:
            raise ValueError(
                "tenant_id is required in payload -- cannot insert without isolation field"
            )
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=chunk_id, vector=vector, payload=payload)],
        )
```

The `build_tenant_filter` uses Qdrant `must` (AND) conditions for tenant isolation and draft exclusion -- these are non-negotiable constraints. The access level check uses `should` (OR) with `minimum_should_match=1` so a user with "public" and "internal" access sees both categories without duplicating filter logic. The `upsert_chunk` method raises immediately if `tenant_id` is missing -- a missing isolation field is a security bug, not a data quality issue, and it must fail loudly.

---

### Architecture Layer

#### Multi-Tenant RAG Architecture

```
[User HTTP Request]
         |
         v
[Auth Middleware]
  validate JWT token
  extract: user_id, tenant_id (from JWT claim -- never from body)
  build UserContext: groups, access_levels
         |
         v
[Query Embedding]
  embed user query
         |
         v
[TenantAwareVectorStore.search()]
  Qdrant: payload filter at HNSW graph traversal
  Filter: tenant_id = user.tenant_id
          AND is_draft = false
          AND access_level in user.access_levels
         |
         v (only user's allowed documents visible)
[Context Assembly + LLM]
         |
         v
[Audit Logger]
  log: user_id, tenant_id, query_hash, chunk_ids_returned, timestamp
  DO NOT log raw query text (may contain PII)
```

#### Isolation Failure Modes and Defenses

```
Failure 1: Missing tenant filter
  Symptom: search returns results from all tenants
  Defense: unit test asserting cross-tenant results never returned
           linter rule requiring build_tenant_filter() before every search call

Failure 2: Tenant ID from user-supplied input
  Symptom: user forges a different tenant_id in request body
  Defense: ALWAYS extract tenant_id from validated JWT, never from request body

Failure 3: Deleted documents still in index
  Symptom: deleted documents appear in results
  Defense: hard-delete from index OR set is_deleted=True and add to filter

Failure 4: Wrong collection for tenant
  Symptom: data inserted into wrong collection
  Defense: derive collection name programmatically from tenant_id, never accept as parameter
```

> **Bridge:** Your access control is locked down. But access control only helps if the documents in the index are current. An isolated index full of outdated documents gives perfectly confident wrong answers. Index freshness is the next problem.

---

⚡ **Senior Checklist -- 3.12**
- [ ] Never derive tenant_id from request body -- always extract from validated JWT claim; this is the most common access control vulnerability in RAG systems
- [ ] Apply tenant filter at the index scan level (Qdrant payload filter or pgvector RLS), not in post-retrieval application code
- [ ] Write a cross-tenant isolation integration test and run it in CI -- assert tenant B's documents never appear in tenant A's results
- [ ] Log chunk_ids returned per query for audit trails but never log raw query text (PII risk)
- [ ] Hard-delete or mark `is_deleted=True` on document removal -- missing this means deleted documents keep surfacing in answers indefinitely
- [ ] Test the empty-results case: when a user has no matching documents, return "no information found" gracefully rather than a 500 error
- [ ] Add namespace isolation on top of payload filters for regulated workloads -- payload filters can have bugs; namespaces are structural and cannot

---

## 3.13 Index Freshness

### 🧠 Mental Model
> An index that does not update is a book that is never reprinted. The first edition was accurate when printed. By the time users read it, the facts have changed -- but the book confidently presents the old version. Index freshness is the discipline of keeping the book up to date before users trust it.

**Connects to:** Access Control (3.12) -> **Index Freshness** -> RAG Evaluation (3.14)
**Parent concept:** Data pipeline reliability
**Builds on:** Section 3.8 -- metadata contains `updated_at` and `expires_at`; index freshness is the system that keeps those values accurate and removes stale content

---

### Concept Layer

#### Why Stale Indexes Are Dangerous

A stale index fails invisibly. There is no error, no warning, no alert. The system returns results confidently -- describing a world that no longer exists. If your pricing policy changed last week but the index still has the old policy, every customer who asks "what is the price?" gets the old answer, presented with full confidence.

Three types of staleness -- each harder to detect than the last:

**Type 1: New documents not yet indexed.** Someone uploaded a new policy PDF but the pipeline has not processed it yet. Queries that should find this document get no results, or fall back to older documents. The user assumes the information does not exist.

**Type 2: Updated documents not re-indexed.** A document was changed but the old version is still in the index. Queries return the outdated version. The model cites it as authoritative. This is the most dangerous type: the answer is wrong AND confident.

**Type 3: Deleted documents still in index.** A document was removed from the source but vectors remain in the database. Queries still surface it. If the document was removed for a reason -- regulatory compliance, product discontinuation, legal hold -- this is a serious problem.

**The rule:** The index is only as fresh as the last time your ingestion pipeline ran. If you do not know when that was, you do not know how fresh your answers are.

#### Change Detection Strategies

**Strategy 1: Polling (scheduled re-scan).**
Every N minutes, scan the document source for files modified since the last scan timestamp.

```
Every 15 minutes:
  for each document in source:
    if doc.modified_at > last_indexed_at[doc.id]:
      re-embed and upsert to vector store
      update last_indexed_at[doc.id] = now
```

Simple to implement. Maximum staleness = polling interval (15 min). Works for most enterprise deployments.

**Strategy 2: Webhooks (event-driven).**
The document source (SharePoint, Confluence, S3) sends an HTTP event when a document changes. The ingestion pipeline processes it immediately.

```
Document updated in SharePoint
  -> webhook fires -> SQS/Pub-Sub message queue
  -> ingestion worker: re-embed and upsert
  -> index updated within seconds
```

Near-real-time freshness. Requires the source system to support webhooks. More complex but appropriate for critical, frequently-changing content.

**Strategy 3: Content hash comparison.**
Store a SHA-256 hash of each document's content at index time. On each poll, re-hash the document and compare. Re-index only if the hash changed. This catches content changes even when modification timestamps are unreliable -- some storage systems update `mtime` when a file is simply opened for reading, not just written.

Think of it like a fingerprint. The fingerprint only changes if the actual content changes. Timestamps change for all sorts of reasons that have nothing to do with the content.

**In short:** Use polling for simplicity, webhooks for real-time freshness, content hashes to catch updates that timestamps miss.

#### Tombstoning Deleted Documents

When a document is deleted from the source, you cannot just leave its vectors in the index. You must actively remove them -- and deletions are easy to miss in a polling pipeline that only looks for new and changed files.

**Approach 1: Reconciliation scan.** Periodically compare the set of document IDs in the index against the set in the source. Any ID in the index but missing from the source is a deleted document. Remove its chunks.

**Approach 2: Soft-delete.** When deletion is detected, set `is_deleted=True` in the index payload. Add `is_deleted != true` to every retrieval filter. Schedule a cleanup job to hard-delete soft-deleted chunks after 24 hours.

Soft-delete is safer because the filter kicks in before the hard delete completes. Even if deletion and filter propagation happen slightly out of order, the document is already invisible to queries. You never have a gap where a just-deleted document leaks into answers.

**In short:** Deletions are the hardest part of index freshness. Run reconciliation scans to catch deletions that event-driven pipelines miss.

---

### Engineering Layer

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol


@dataclass
class DocumentRecord:
    doc_id: str
    source_path: str
    content_hash: str       # SHA-256 of raw content bytes
    last_indexed_at: str    # ISO 8601 timestamp
    chunk_ids: list[str]    # all chunk IDs produced from this document


class DocumentSource(Protocol):
    def list_documents(self) -> list[dict]:
        ...  # returns [{doc_id, path, modified_at}]

    def get_content(self, doc_id: str) -> bytes:
        ...


class IndexRegistry:
    def __init__(self) -> None:
        self._records: dict[str, DocumentRecord] = {}

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._records.get(doc_id)

    def upsert(self, record: DocumentRecord) -> None:
        self._records[record.doc_id] = record

    def all_indexed_ids(self) -> set[str]:
        return set(self._records.keys())

    def mark_deleted(self, doc_id: str) -> list[str]:
        record = self._records.pop(doc_id, None)
        return record.chunk_ids if record else []


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def run_freshness_sync(
    doc_source: DocumentSource,
    registry: IndexRegistry,
    on_upsert: Callable[[str, bytes], list[str]],    # (doc_id, content) -> new chunk_ids
    on_delete_chunks: Callable[[list[str]], None],   # (chunk_ids) -> None
) -> dict[str, int]:
    source_docs = {d["doc_id"]: d for d in doc_source.list_documents()}
    source_ids = set(source_docs.keys())
    indexed_ids = registry.all_indexed_ids()

    new_docs = source_ids - indexed_ids
    deleted_docs = indexed_ids - source_ids
    candidate_updates = source_ids & indexed_ids

    stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}

    for doc_id in new_docs:
        content = doc_source.get_content(doc_id)
        chunk_ids = on_upsert(doc_id, content)
        registry.upsert(DocumentRecord(
            doc_id=doc_id,
            source_path=source_docs[doc_id]["path"],
            content_hash=sha256(content),
            last_indexed_at=datetime.now(timezone.utc).isoformat(),
            chunk_ids=chunk_ids,
        ))
        stats["added"] += 1

    for doc_id in candidate_updates:
        content = doc_source.get_content(doc_id)
        new_hash = sha256(content)
        record = registry.get(doc_id)
        if record and record.content_hash == new_hash:
            stats["skipped"] += 1
            continue  # content unchanged -- skip expensive re-embedding
        if record:
            on_delete_chunks(record.chunk_ids)  # remove old chunks first
        chunk_ids = on_upsert(doc_id, content)
        registry.upsert(DocumentRecord(
            doc_id=doc_id,
            source_path=source_docs[doc_id]["path"],
            content_hash=new_hash,
            last_indexed_at=datetime.now(timezone.utc).isoformat(),
            chunk_ids=chunk_ids,
        ))
        stats["updated"] += 1

    for doc_id in deleted_docs:
        chunk_ids = registry.mark_deleted(doc_id)
        on_delete_chunks(chunk_ids)
        stats["deleted"] += 1

    return stats
```

The content hash comparison skips re-embedding when only the document's metadata changes (like a rename) but the content does not. In typical enterprise document stores, 70-90% of documents are unchanged on any given poll -- skipping their re-embedding reduces pipeline cost by the same factor. The three-phase sync (new, updated, deleted) explicitly handles deletion -- which most naive polling pipelines miss entirely, allowing deleted documents to live in the index indefinitely.

---

### Architecture Layer

#### Index Freshness Pipeline

```
[Document Sources: S3, SharePoint, Confluence, local FS]
         |
         |-- Webhook events -----> [Message Queue] ---> [Ingestion Worker]
         |                                                       |
         |-- Polling (every 15m) -> [Freshness Sync] -----------+
                                                                |
                                                    [Content Hash Compare]
                                                      changed? -> re-embed
                                                      unchanged? -> skip
                                                                |
                                                    [Vector DB Upsert]
                                                      delete old chunk_ids
                                                      insert new chunk_ids
                                                                |
                                                    [Index Registry Update]
                                                      new hash + new chunk_ids
```

#### Recommended Freshness Schedule

| Frequency | Trigger | Purpose |
|-----------|---------|---------|
| Real-time | Webhook event | Process single document change immediately |
| Every 15 min | Scheduler | Catch missed events, process new docs |
| Every 24 hours | Daily job | Full reconciliation: find phantom deletions |
| Every 7 days | Weekly job | Validate index integrity, detect corruption |

| Freshness risk | Mitigation | Cost |
|---------------|-----------|------|
| New doc not indexed | Webhook + 15-min polling | Low |
| Updated doc, old vectors | Content hash on every poll | Low |
| Deleted doc still in index | Daily reconciliation scan | Low |
| Index corruption | Weekly integrity check | Medium |

> **Bridge:** Your index is fresh, isolated, and well-structured. But how do you know if it actually answers questions correctly? Feelings are not metrics. RAGAS gives you the numbers you need to make that judgment.

---

⚡ **Senior Checklist -- 3.13**
- [ ] Always use content hashes (SHA-256), not just modification timestamps -- some storage systems update mtime on read, not just write
- [ ] Delete old chunks before inserting new ones for updated documents -- just upserting leaves orphaned old chunk IDs in your registry
- [ ] Run a daily reconciliation scan comparing index IDs to source IDs -- this is the only way to catch deletions in a polling-only pipeline
- [ ] Alert if `last_sync_at` is more than 30 minutes stale -- a broken ingestion pipeline means invisible outdated answers
- [ ] Store `chunk_ids` per document in the registry -- you cannot delete the right chunks without knowing which ones belong to which document
- [ ] Test the deletion pipeline explicitly: insert a document, delete it from source, run sync, verify it is absent from query results
- [ ] Use soft-delete with 24-hour cleanup for regulated industries that require an audit trail before hard deletion

---

## 3.14 RAG Evaluation

### 🧠 Mental Model
> "It seems to work" is not a production metric. RAGAS gives you four numbers that measure whether your RAG system actually retrieves the right content and generates faithful answers. Those four numbers tell you exactly which part of your pipeline is broken and by how much.

**Connects to:** Index Freshness (3.13) -> **RAG Evaluation** -> Failure Modes (3.15)
**Parent concept:** Measurement and quality assurance
**Builds on:** Section 3.9 -- retrieval patterns determine context quality; this section measures whether that context produces correct answers

---

### Concept Layer

#### Why You Need Dedicated RAG Metrics

Standard software metrics (uptime, latency, error rate) tell you whether the system is running, not whether it is answering correctly. A RAG system can be 100% available, respond in 200ms, and return zero errors -- while giving completely wrong answers to 40% of queries. You need metrics that measure answer quality, not just system health.

RAGAS (Retrieval-Augmented Generation Assessment) is the standard framework for this. It decomposes RAG quality into four distinct scores, each measuring a different potential failure point.

Think of it like diagnosing a car that gives wrong directions. The four RAGAS metrics answer: Did the GPS find the right starting location? Did the GPS find the right destination? Did the driver follow the GPS? Did the GPS directions actually make sense?

#### The Four RAGAS Metrics

**Metric 1: Context Recall (0.0-1.0)**
"Did retrieval find all the information needed to answer the question?"

This measures whether the retrieved chunks contain all the information present in the reference (correct) answer. If the correct answer requires three facts and your retrieved chunks only contain two of them, context recall is roughly 0.67.

Think of it like packing for a trip. Context recall measures whether you remembered to pack everything you needed -- not whether you packed things you did not need.

**A score below 0.7 means your retrieval is missing information.** Fix: better chunking, larger k, query expansion.

**Metric 2: Context Precision (0.0-1.0)**
"Of everything retrieved, how much was actually relevant?"

This measures the fraction of retrieved chunks that are actually useful for answering the question. Retrieving 10 chunks where 8 are irrelevant gives a context precision of 0.2, even if the 2 relevant ones are perfect.

Think of it like a search result page. Context precision measures the signal-to-noise ratio. A page full of irrelevant results has low precision even if the right answer is buried on page 3.

**A score below 0.7 means your retrieval returns too much noise.** Fix: raise similarity threshold, add metadata filters, use reranking.

**Metric 3: Faithfulness (0.0-1.0)**
"Does the model's answer only contain claims that are grounded in the retrieved context?"

This measures whether the model stayed within the bounds of what it was told. An answer that adds facts not present in the retrieved chunks is unfaithful -- the model hallucinated or used its training data instead of the provided context.

A faithfulness score of 0.85 means 85% of claims in the answer can be traced to the retrieved chunks. The remaining 15% were invented.

**A score below 0.8 means the model is hallucinating.** Fix: stronger source-constraint prompt, higher confidence threshold, better chunk quality.

**Metric 4: Answer Relevancy (0.0-1.0)**
"Does the answer actually address what was asked?"

This measures whether the generated answer is on-topic and complete. An answer that is fully grounded in sources but answers a different question scores low on answer relevancy.

Think of it as the conversation partner test: if someone asked "what is your refund policy?" and you answered "we have a great support team available 24/7", you were grounded and truthful -- but you did not answer the question.

**A score below 0.8 means answers are off-topic or incomplete.** Fix: improve system prompt focus, adjust retrieval to better match query intent.

#### Reading the Scores Together

```
High context recall + low context precision:
  Retrieving broadly but noisily -- use reranking and metadata filters

Low context recall + high context precision:
  Retrieval is precise but incomplete -- use query expansion or hybrid search

High context scores + low faithfulness:
  Retrieval is good but model is hallucinating -- strengthen prompt constraints

High everything except answer relevancy:
  All components work but answers miss the point -- fix system prompt and query parsing
```

**In short:** RAGAS is a four-dimensional compass for RAG quality. Each number points to a specific failure mode. Low scores in specific metrics tell you exactly what to fix.

---

### Engineering Layer

```python
from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)


@dataclass
class EvaluationSample:
    question: str
    answer: str                   # the RAG system's generated answer
    contexts: list[str]           # the retrieved chunk texts
    ground_truth: str             # the known correct answer


def build_ragas_dataset(samples: list[EvaluationSample]) -> Dataset:
    return Dataset.from_dict({
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples],
    })


def run_ragas_evaluation(samples: list[EvaluationSample]) -> dict[str, float]:
    dataset = build_ragas_dataset(samples)
    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    return {
        "context_precision": round(results["context_precision"], 3),
        "context_recall": round(results["context_recall"], 3),
        "faithfulness": round(results["faithfulness"], 3),
        "answer_relevancy": round(results["answer_relevancy"], 3),
        "ragas_score": round(
            (results["context_precision"]
             + results["context_recall"]
             + results["faithfulness"]
             + results["answer_relevancy"]) / 4,
            3,
        ),
    }


def generate_evaluation_samples(
    rag_pipeline,                           # callable: (question) -> (answer, contexts)
    test_set: list[dict[str, str]],         # [{"question": ..., "ground_truth": ...}]
) -> list[EvaluationSample]:
    samples: list[EvaluationSample] = []
    for item in test_set:
        answer, contexts = rag_pipeline(item["question"])
        samples.append(EvaluationSample(
            question=item["question"],
            answer=answer,
            contexts=contexts,
            ground_truth=item["ground_truth"],
        ))
    return samples


def evaluate_rag_system(
    rag_pipeline,
    test_set: list[dict[str, str]],
    score_thresholds: dict[str, float] | None = None,
) -> dict[str, float | bool]:
    if score_thresholds is None:
        score_thresholds = {
            "context_precision": 0.70,
            "context_recall": 0.70,
            "faithfulness": 0.80,
            "answer_relevancy": 0.80,
        }

    samples = generate_evaluation_samples(rag_pipeline, test_set)
    scores = run_ragas_evaluation(samples)

    scores["passed"] = all(
        scores[metric] >= threshold
        for metric, threshold in score_thresholds.items()
    )
    return scores
```

The `evaluate_rag_system` function wraps the full evaluation loop into a single callable with configurable pass/fail thresholds. The `passed` field makes it usable as a CI gate: if `faithfulness < 0.80` or `context_recall < 0.70`, the evaluation fails and blocks a deployment. Running this on 100 test questions takes 3-5 minutes with a Claude or OpenAI backend and gives you objective, reproducible quality metrics at every release.

---

### Architecture Layer

#### RAG Evaluation Architecture

```
[Test Question Set]
  (100-500 questions with known correct answers)
         |
         v
[RAG Pipeline Under Test]
  for each question:
    retrieve context chunks
    generate answer
         |
         v
[RAGAS Evaluator]
  context_precision:  how much retrieved content was relevant?
  context_recall:     how much needed content was retrieved?
  faithfulness:       how much of the answer is grounded in context?
  answer_relevancy:   how well does the answer address the question?
         |
         v
[Evaluation Report]
  scores per metric
  per-question breakdown (which questions failed?)
  pass/fail decision
```

#### Evaluation Cadence Recommendations

| When to run | Test set size | Purpose |
|------------|--------------|---------|
| Every PR/deployment | 50-100 questions | Catch regressions before production |
| Weekly | 200-500 questions | Trend analysis, catch slow drift |
| After embedding model change | 500+ questions | Validate no quality regression |
| After chunking strategy change | 500+ questions | Validate improvement claimed |

#### Building a Good Test Set

RAGAS requires a test set of questions with known correct answers. How to build one:

1. **Sample real queries** from production query logs (anonymized).
2. **Write reference answers** manually or with a stronger model (GPT-4o or Claude Opus).
3. **Include hard cases**: questions that require combining information from multiple chunks, questions at the boundary of your knowledge base, questions the system should decline to answer.
4. **Minimum viable test set**: 50 questions covers basic regression detection. 200 questions gives statistically reliable trend detection.

> **Bridge:** You now have metrics. Low scores point to specific failure modes. The next section catalogs exactly what each failure looks like, why it happens, and how to diagnose it from your RAGAS numbers.

---

⚡ **Senior Checklist -- 3.14**
- [ ] Build a test set of 50-200 questions with ground-truth answers before launch -- you cannot measure what you cannot test
- [ ] Set RAGAS score thresholds as deployment gates: context_recall >= 0.70, faithfulness >= 0.80 -- block deploys that fail
- [ ] Run evaluation after every chunking strategy or embedding model change -- never assume a change improved quality without measuring
- [ ] Log per-question breakdown, not just averages -- a 0.85 average can hide 20 catastrophically wrong answers
- [ ] Include "out of scope" questions in your test set -- faithfulness must be high for questions the system should decline
- [ ] Re-evaluate on a fresh sample each week -- averages on a fixed test set can be gamed by overfitting; fresh samples reveal drift
- [ ] Track RAGAS score trends over time, not just point-in-time values -- a gradual decline in context_recall signals index staleness


## 3.15 RAG Failure Modes

### 🧠 Mental Model
> Every RAG failure traces back to one of five root causes: bad chunking, wrong retrieval, hallucination despite good retrieval, context overflow, or semantic drift. RAGAS tells you the score. This section tells you which score maps to which root cause and exactly what to do about it.

**Connects to:** RAG Evaluation (3.14) -> **Failure Modes** -> Advanced RAG (3.16)
**Parent concept:** Production debugging and quality improvement
**Builds on:** Section 3.14 -- RAGAS scores quantify failures; this section diagnoses the mechanism behind each score

---

### Concept Layer

#### Failure Mode 1: Retrieval Misses (Low Context Recall)

**What it looks like:** The system says "I do not have information about that" even though the answer is in your knowledge base. Or it gives a partial answer missing key facts.

**Why it happens:** The query vector and the relevant document vector are not close enough in embedding space. This can happen because:
- The query uses different vocabulary than the document ("cancel" vs "terminate subscription")
- The chunk is too small and lacks enough context for the embedding to be meaningful
- The relevant information is spread across multiple chunks and no single chunk covers the full answer
- The embedding model was not trained on your domain (medical queries against a general-purpose embedder)

**RAGAS signal:** `context_recall < 0.70`

**How to fix:**
1. Add BM25 hybrid search -- it finds exact term matches that dense retrieval misses
2. Increase chunk size (512 -> 1024 tokens) -- larger chunks have more context for the embedder
3. Use query expansion (section 3.16) -- generate multiple phrasings of the query
4. Switch to a domain-specific embedding model

#### Failure Mode 2: Retrieval Noise (Low Context Precision)

**What it looks like:** The model gives answers that mix correct information with incorrect or irrelevant information, citing multiple sources confusingly.

**Why it happens:** The retrieval returns too many marginally-relevant chunks that confuse the model. The model cannot distinguish highly-relevant from loosely-relevant context and attempts to synthesize all of it.

**RAGAS signal:** `context_precision < 0.70`

**How to fix:**
1. Raise the similarity score threshold (0.70 -> 0.80)
2. Add a cross-encoder reranker -- it filters the top-20 down to a focused top-5
3. Add metadata filters to restrict the document type or date range
4. Reduce k (retrieve 5 instead of 10) and accept lower recall in exchange for higher precision

#### Failure Mode 3: Hallucination Despite Good Retrieval (Low Faithfulness)

**What it looks like:** RAGAS shows high context recall and context precision (retrieval is working), but the model's answers contain facts not present in the retrieved chunks.

**Why it happens:** The model is using its parametric memory — facts baked in during training — instead of sticking to the provided context. This happens when:
- The source constraint in the system prompt is too weak ("use the context below" vs "ONLY use the information in <context>")
- The question is on a topic the model knows well from training -- it substitutes its own knowledge
- The context chunks contain the right information but are formatted poorly (tables, code blocks, numbers)

**RAGAS signal:** `faithfulness < 0.80` with `context_recall >= 0.70`

**How to fix:**
1. Strengthen the source constraint: "Answer using ONLY the information in <context>. If the answer is not there, say so explicitly."
2. Use temperature 0.0 for factual RAG -- reduces creative generation
3. Add explicit test for the "I do not have that information" path -- hallucination often happens on boundary cases
4. Format numbers and key facts prominently in chunks so the model cannot miss them

#### Failure Mode 4: Context Overflow (Silent Truncation)

**What it looks like:** Answers that miss information present in the knowledge base, despite high RAGAS scores on smaller test queries. The failure is inconsistent -- works for short queries, fails for long ones.

**Why it happens:** The assembled context exceeds the context window and the end is silently truncated. The model answers based on incomplete context without indicating anything is missing.

**RAGAS signal:** Not detected by RAGAS (it evaluates what was assembled, not what was truncated). Detected by: logging `truncated=True` events in your context assembly layer.

**How to fix:**
1. Add token counting before assembly (section 3.10) -- hard requirement
2. Set `truncated=True` flag and log it -- alert when rate exceeds 5% of queries
3. Use Matryoshka truncation to reduce embedding dimensions and fit more chunks
4. Reduce chunk size (1024 -> 512) so more chunks fit in budget

#### Failure Mode 5: Semantic Drift (Gradual Quality Decline)

**What it looks like:** RAGAS scores that were acceptable at launch decline slowly over 2-3 months. Queries that worked before start returning worse answers.

**Why it happens:** Your documents change but the index does not keep up. New terminology is introduced in updated documents, but the index still has old embeddings. The gap between the vocabulary of user queries and the vocabulary of indexed documents widens over time.

**RAGAS signal:** Gradual decline in `context_recall` over weeks, not a sudden drop.

**How to fix:**
1. Check index freshness: when was the last ingestion run? (Section 3.13)
2. Check if user query vocabulary has shifted (new product names, new terminology)
3. Re-evaluate embedding model on current documents -- the model that was best 6 months ago may have been surpassed
4. Run a full re-embedding of the index if vocabulary drift is confirmed

**In short:** Five failure modes, five root causes. RAGAS scores tell you which one you have. Each one has a specific fix. None of them require guessing.

---

### Engineering Layer

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureMode(Enum):
    RETRIEVAL_MISS = "retrieval_miss"          # context_recall < 0.70
    RETRIEVAL_NOISE = "retrieval_noise"        # context_precision < 0.70
    HALLUCINATION = "hallucination"            # faithfulness < 0.80
    CONTEXT_OVERFLOW = "context_overflow"     # truncated=True in assembly
    SEMANTIC_DRIFT = "semantic_drift"         # gradual recall decline over time


@dataclass
class DiagnosisResult:
    primary_failure: FailureMode | None
    secondary_failures: list[FailureMode]
    recommended_fixes: list[str]
    severity: str   # "critical" | "moderate" | "minor"


def diagnose_rag_failure(
    ragas_scores: dict[str, float],
    truncation_rate: float = 0.0,     # fraction of queries where context was truncated
    recall_trend: float = 0.0,        # change in context_recall over last 30 days (negative = declining)
) -> DiagnosisResult:
    failures: list[FailureMode] = []
    fixes: list[str] = []

    if ragas_scores.get("context_recall", 1.0) < 0.70:
        failures.append(FailureMode.RETRIEVAL_MISS)
        fixes.extend([
            "Add BM25 hybrid search to catch exact-term matches",
            "Increase chunk size from 512 to 1024 tokens",
            "Implement query expansion (HyDE or multi-query)",
        ])

    if ragas_scores.get("context_precision", 1.0) < 0.70:
        failures.append(FailureMode.RETRIEVAL_NOISE)
        fixes.extend([
            "Add cross-encoder reranker to filter top-20 down to top-5",
            "Raise similarity score threshold from 0.70 to 0.80",
            "Reduce k from 10 to 5 and tighten metadata filters",
        ])

    if ragas_scores.get("faithfulness", 1.0) < 0.80:
        failures.append(FailureMode.HALLUCINATION)
        fixes.extend([
            "Strengthen source constraint: 'Answer using ONLY the <context> block'",
            "Set temperature=0.0 for factual RAG queries",
            "Test 'out of scope' path: does system say 'I don't know' correctly?",
        ])

    if truncation_rate > 0.05:
        failures.append(FailureMode.CONTEXT_OVERFLOW)
        fixes.extend([
            "Add explicit token counting before context assembly",
            "Reduce chunk size or k to fit within token budget",
            "Use Matryoshka embedding truncation to 512 dims for more chunks",
        ])

    if recall_trend < -0.05:  # context_recall dropped >5 points over 30 days
        failures.append(FailureMode.SEMANTIC_DRIFT)
        fixes.extend([
            "Check ingestion pipeline: when did it last run?",
            "Run freshness sync and re-evaluate after indexing new documents",
            "Consider re-embedding full index if vocabulary drift is confirmed",
        ])

    if not failures:
        return DiagnosisResult(
            primary_failure=None,
            secondary_failures=[],
            recommended_fixes=["No critical failures detected. Monitor trends."],
            severity="minor",
        )

    severity = "critical" if FailureMode.HALLUCINATION in failures else "moderate"

    return DiagnosisResult(
        primary_failure=failures[0],
        secondary_failures=failures[1:],
        recommended_fixes=fixes,
        severity=severity,
    )
```

The `diagnose_rag_failure` function maps RAGAS score thresholds to specific failure modes and recommended fixes. Hallucination is treated as "critical" severity regardless of other scores because it means users are receiving confidently wrong information -- which is worse than no information at all. The `recall_trend` parameter detects semantic drift specifically, which point-in-time scores cannot reveal.

---

### Architecture Layer

#### Failure Diagnosis Decision Tree

```
[RAGAS Evaluation Results]
         |
         v
context_recall < 0.70?
   YES -> Retrieval Miss
         Fixes: hybrid search, larger chunks, query expansion
   NO  -> continue
         |
         v
context_precision < 0.70?
   YES -> Retrieval Noise
         Fixes: reranker, higher threshold, reduce k
   NO  -> continue
         |
         v
faithfulness < 0.80?
   YES -> Hallucination
         Fixes: stronger source constraint, temperature=0.0
   NO  -> continue
         |
         v
truncation_rate > 5%?
   YES -> Context Overflow
         Fixes: token counting, smaller chunks, lower k
   NO  -> continue
         |
         v
context_recall declining week-over-week?
   YES -> Semantic Drift
         Fixes: check ingestion, re-embed, freshness sync
   NO  -> System healthy. Monitor.
```

| Failure mode | RAGAS indicator | Root cause | Primary fix |
|-------------|----------------|-----------|------------|
| Retrieval miss | context_recall < 0.70 | Semantic gap between query and doc | Hybrid search + query expansion |
| Retrieval noise | context_precision < 0.70 | Too many irrelevant chunks | Reranker + threshold tuning |
| Hallucination | faithfulness < 0.80 | Model ignoring context | Stronger source constraint |
| Context overflow | truncated=True > 5% | Token budget exceeded | Token counting + smaller chunks |
| Semantic drift | Recall declining over weeks | Index stale, vocab shifted | Freshness sync + re-embedding |

> **Bridge:** You can now diagnose failures. The next question is: what advanced techniques exist to push quality past the baseline? Query expansion, HyDE, and multi-hop retrieval are the tools that get you from "acceptable" to "excellent."

---

⚡ **Senior Checklist -- 3.15**
- [ ] Map every RAGAS score below threshold to a specific failure mode before making any changes -- treat it like debugging, not guessing
- [ ] Distinguish hallucination (faithfulness < 0.80) from retrieval miss (context_recall < 0.70) -- they have opposite fixes
- [ ] Log truncation_rate separately from RAGAS -- context overflow is invisible to RAGAS and requires its own monitoring
- [ ] Track context_recall as a weekly trend, not just a point-in-time score -- gradual decline is the semantic drift signal
- [ ] Fix faithfulness failures first -- a confident wrong answer is worse than a correct "I do not have that information"
- [ ] Never apply multiple fixes simultaneously -- you will not know which one worked; fix one, re-evaluate, then proceed
- [ ] Include "out of scope" queries in your test set and verify faithfulness score on those -- hallucination is most likely on boundary cases

---

## 3.16 Advanced RAG Patterns

### 🧠 Mental Model
> Baseline RAG is "embed the query, find similar chunks." Advanced RAG is "think harder about what to look for before looking." Query expansion, HyDE, and multi-hop retrieval each attack a different weakness of single-pass dense retrieval. Use them when baseline RAG has a specific measurable failure.

**Connects to:** Failure Modes (3.15) -> **Advanced RAG** -> Hybrid Architectures (3.17)
**Parent concept:** Retrieval quality improvement strategies
**Builds on:** Section 3.9 -- retrieval patterns; this section extends them with higher-level strategies that improve retrieval quality before similarity search runs

---

### Concept Layer

#### Pattern 1: Query Expansion

**The problem:** A user query is typically 5-15 words. The relevant document may use completely different words to describe the same concept. Single-query retrieval misses these vocabulary mismatches.

**The solution:** Generate multiple phrasings of the same query and retrieve for each one, then merge the results.

```
User query: "can I return my order"

Expanded queries:
  1. "can I return my order"           (original)
  2. "what is the refund policy?"      (reformulation)
  3. "how do I get a refund?"          (intent expansion)
  4. "return and exchange process"     (topic expansion)

Retrieve top-5 for each -> merge with deduplication -> rerank combined pool
```

Think of it like searching a library by looking under multiple related catalog entries instead of just one. You are more likely to find what you need even if the librarian filed it under a slightly different heading.

**Cost:** One extra LLM call to generate expanded queries (fast and cheap with Haiku at $0.00025/1K tokens). Retrieval cost multiplies by the number of queries.
**Quality gain:** 10-20% improvement in context_recall on vocabulary-mismatch queries.

#### Pattern 2: HyDE (Hypothetical Document Embeddings)

**The problem:** Short queries (5-15 words) produce less informative embeddings than full documents. The query vector is in a different part of the embedding space than the answer vector -- even though they should be related.

**The solution:** Before retrieving, ask the LLM to write a hypothetical document that would answer the query. Then embed THAT document and use it as the query vector.

```
User query: "what is the refund policy for digital products?"

Step 1 -- Generate hypothetical answer:
LLM: "Digital products are non-refundable after download unless the file
      is corrupt or does not match the description. Refund requests must
      be submitted within 48 hours of purchase."

Step 2 -- Embed the hypothetical answer (not the query)
Step 3 -- Search with the hypothetical answer embedding
```

The hypothetical answer is in the same region of embedding space as real policy documents -- because both are policy-style text. The original query is in "question space", which is farther from "answer space."

This is like searching for a book by describing what the book would contain, rather than asking a question about the topic. Librarians find it easier to match book descriptions than questions.

**Cost:** One LLM call to generate the hypothetical answer (20-50 tokens, fast).
**Quality gain:** 5-15% improvement in context_recall for complex or technical queries.
**When NOT to use:** Simple, well-phrased queries where the query embedding is already informative. HyDE adds latency and cost; only use it when baseline fails.

#### Pattern 3: Multi-Hop Retrieval (Iterative Retrieval)

**The problem:** Some questions require combining information from multiple documents that are not co-located in the index. "Compare the refund policies for digital vs physical products" requires finding two separate document sections, each of which is only partially relevant.

**The solution:** Retrieve once, read the results, then decide whether to retrieve again with a more targeted follow-up query.

```
Query: "what documents do I need to apply for a visa and how long does it take?"

Hop 1: retrieve on "visa application documents required"
  -> finds: list of required documents

Hop 2: use documents from hop 1 to form better query:
  "processing time for tourist visa with [passport + bank statement + invitation]"
  -> finds: processing time information for that specific document combination

Combine both hops -> assemble full context -> answer
```

Think of it like a research project. You read one paper, find a reference to a related concept, then go find that second paper. Each hop gets you closer to a complete answer.

**Cost:** N extra retrieval rounds + N extra LLM calls (one to decide whether to hop and what to search next).
**Quality gain:** Critical for multi-part questions. For single-hop questions, adds latency with no benefit.
**When to use:** When your RAGAS `context_recall` is low specifically on complex, multi-part queries.

**In short:** Query expansion finds the same answer with different words. HyDE uses an expected answer to find the actual answer. Multi-hop connects answers that span multiple documents.

---

### Engineering Layer

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic


@dataclass
class ExpandedQuery:
    original: str
    expansions: list[str]
    all_queries: list[str]   # original + expansions


def expand_query(
    client: anthropic.Anthropic,
    query: str,
    n_expansions: int = 3,
    model: str = "claude-haiku-4-5-20251001",  # cheap model is sufficient
) -> ExpandedQuery:
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n_expansions} alternative phrasings of this query "
                f"that preserve the meaning but use different vocabulary.\n"
                f"Query: {query}\n\n"
                f"Return only the phrasings, one per line, no numbering."
            ),
        }],
    )
    lines = [line.strip() for line in response.content[0].text.strip().split("\n") if line.strip()]
    expansions = lines[:n_expansions]
    return ExpandedQuery(
        original=query,
        expansions=expansions,
        all_queries=[query] + expansions,
    )


def hyde_embed(
    client: anthropic.Anthropic,
    embedder,                    # any object with .embed(text) -> EmbeddingResult
    query: str,
    model: str = "claude-haiku-4-5-20251001",
) -> list[float]:
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "Write a short passage (2-3 sentences) that would directly answer "
                f"this question, as if it were from an official document:\n\n{query}"
            ),
        }],
    )
    hypothetical_doc = response.content[0].text.strip()
    return embedder.embed(hypothetical_doc).vector


def multi_hop_retrieve(
    client: anthropic.Anthropic,
    vector_store,                   # .search(embedding, user_context, k) -> list[dict]
    embedder,
    user_context,
    initial_query: str,
    max_hops: int = 2,
    k_per_hop: int = 5,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    current_query = initial_query

    for hop in range(max_hops):
        embedding = embedder.embed(current_query).vector
        results = vector_store.search(embedding, user_context, k=k_per_hop)
        all_results.extend(results)

        # Ask LLM whether we have enough information or need another hop
        context_text = "\n".join(r["text"] for r in results[:3])
        assessment = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    f"Original question: {initial_query}\n\n"
                    f"Retrieved context so far:\n{context_text}\n\n"
                    f"Does this context fully answer the question? If not, "
                    f"what specific sub-question should we search for next? "
                    f"Reply: COMPLETE if done, or SEARCH: <follow-up query>"
                ),
            }],
        )
        response_text = assessment.content[0].text.strip()
        if response_text.startswith("COMPLETE") or hop == max_hops - 1:
            break
        if response_text.startswith("SEARCH:"):
            current_query = response_text[len("SEARCH:"):].strip()

    # Deduplicate by chunk_id
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in all_results:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            unique.append(r)

    return unique
```

The `expand_query` function uses Claude Haiku (not Sonnet) for expansion because the task is simple and costs 60x less per call. The `hyde_embed` function generates a hypothetical answer and embeds that instead of the query -- bringing the search vector closer to document space. The `multi_hop_retrieve` function uses the LLM to self-assess whether another hop is needed, avoiding unnecessary extra retrievals for simple questions.

---

### Architecture Layer

#### Advanced Retrieval Decision Flow

```
[Incoming Query]
         |
         v
[Query Classifier]
  simple factual? -> baseline dense retrieval
  vocabulary mismatch likely? -> query expansion
  complex/technical? -> HyDE
  multi-part question? -> multi-hop retrieval
         |
         v
[Selected Retrieval Strategy]
  Query expansion: 3-4 parallel retrievals, merge, dedup
  HyDE: generate hypothetical doc, embed it, retrieve
  Multi-hop: retrieve, assess, retrieve again if needed
         |
         v
[Cross-encoder Reranker]
  rerank merged candidate pool
  return top-5
         |
         v
[Context Assembly -> LLM]
```

| Pattern | Latency added | Recall improvement | When to use |
|---------|-------------|-------------------|------------|
| Query expansion | +100ms (1 LLM call) | +10-20% | Vocabulary mismatch, domain jargon |
| HyDE | +150ms (1 LLM call) | +5-15% | Complex technical queries |
| Multi-hop | +300-600ms (2-4 calls) | +20-40% on multi-part | Questions spanning multiple docs |
| All three | +600ms+ | +30-50% | High-stakes enterprise queries |

> **Bridge:** Advanced retrieval patterns improve single-source retrieval. But production systems often have multiple knowledge sources: a vector database, a relational database, a live web search index. Hybrid architectures combine all of them.

---

⚡ **Senior Checklist -- 3.16**
- [ ] Use query expansion only when context_recall is consistently below 0.70 -- it adds latency and cost that simple queries do not need
- [ ] Use Haiku (not Sonnet) for query expansion and HyDE generation -- the task is simple enough that quality difference is negligible
- [ ] Deduplicate results after multi-source retrieval before reranking -- the same chunk from three different queries should count once, not three times
- [ ] Limit multi-hop to max_hops=2 for interactive queries with < 2-second SLA -- each hop adds 300-500ms
- [ ] Test HyDE on your specific query types before deploying -- it helps technical queries but can hurt simple factual queries where the hypothesis is wrong
- [ ] Track per-strategy RAGAS scores separately -- query expansion may help context_recall but hurt context_precision; measure both
- [ ] Cache query expansions for repeated queries -- the same question asked 1,000 times should not cost 1,000 LLM expansion calls

---

## 3.17 Hybrid Architectures

### 🧠 Mental Model
> A pure vector-only RAG system answers from embeddings. A hybrid architecture answers from wherever the right information lives -- vector store for unstructured documents, SQL database for structured records, live search for current events. The router decides which source to hit. The answer comes from the source that knows.

**Connects to:** Advanced RAG (3.16) -> **Hybrid Architectures** -> Enterprise Project (3.18)
**Parent concept:** Multi-source knowledge retrieval
**Builds on:** Sections 3.4-3.9 -- all retrieval patterns; this section extends them to heterogeneous sources beyond a single vector store

---

### Concept Layer

#### Why One Source Is Never Enough

A single vector database works well for document-based knowledge. But real enterprise knowledge is heterogeneous:

- **Unstructured documents** (PDFs, wikis, policies): best served by vector search
- **Structured data** (product catalog, customer records, pricing tables): best served by SQL
- **Real-time information** (stock prices, current events, live inventory): best served by API calls
- **Code** (function signatures, documentation): best served by specialized code search indexes

A system that answers "what is our Q3 revenue?" from a vector store is searching the wrong place -- that answer lives in a database, not a document. A system that tries to answer "what is the current exchange rate?" from indexed documents will always be out of date.

**The rule:** Choose the retrieval source based on the type of information being requested, not just based on which source is easiest to build.

#### Architecture 1: Router-Based Hybrid

A router classifies the query and sends it to the appropriate source.

```
[Query: "what is the price of product SKU-8821?"]
         |
         v
[Query Router -- LLM classifier]
  structured_data? -> SQL database
  document_search? -> vector store
  real_time?       -> API call
         |
         v
[Source-specific retrieval]
  SQL: SELECT price FROM products WHERE sku = 'SKU-8821'
         |
         v
[Answer with result from correct source]
```

This is the cleanest architecture: each query goes to exactly one source, with no blending or merging needed.

**Limitation:** Queries that span multiple sources ("what is the price, and what does the product manual say about maintenance?") require multiple routes and response merging.

#### Architecture 2: Parallel Hybrid (Fan-Out)

All sources are queried simultaneously. Results are merged and reranked before assembly.

```
[Query]
    |
    |----> [Vector store]    -> top-10 document chunks
    |----> [SQL database]    -> structured records
    |----> [Web search API]  -> current web results
    |
    v (all results arrive in parallel)
[Merge and Deduplication]
[Cross-encoder Reranker]
    |
    v
[Top-5 results for context assembly]
```

This approach never misses a relevant source. The tradeoff: SQL records and document chunks arrive in very different formats and must be normalized before the reranker can score them against each other.

**When to use:** High-stakes queries where missing information from any source is unacceptable. The latency is bounded by the slowest source, not the sum.

#### Architecture 3: GraphRAG

Standard RAG retrieves chunks. GraphRAG builds a **knowledge graph** from your documents and traverses it at query time.

How it works:
1. During ingestion, extract entities (people, products, policies) and relationships from documents
2. Store these as a graph (nodes = entities, edges = relationships)
3. At query time, identify relevant graph nodes, then retrieve the chunks associated with those nodes

Why it helps: Multi-hop reasoning ("which employees approved the policy that applies to international orders?") is natural in a graph but requires multi-hop retrieval in a vector store.

Think of it like the difference between a dictionary and an encyclopedia. A dictionary looks up words in isolation. An encyclopedia has cross-references that connect related articles. GraphRAG is the encyclopedia approach.

**Cost:** Significantly more complex to build and maintain. Use it only when your queries regularly require combining information across multiple linked entities.

**In short:** Router-based hybrid sends queries to the right source. Fan-out hybrid queries all sources in parallel. GraphRAG traverses entity relationships. Each adds capability at the cost of complexity.

---

### Engineering Layer

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import anthropic


class RetrievalSource(Enum):
    VECTOR_STORE = "vector_store"
    SQL_DATABASE = "sql_database"
    WEB_SEARCH = "web_search"
    API_CALL = "api_call"


@dataclass
class HybridResult:
    text: str
    score: float
    source: RetrievalSource
    metadata: dict[str, Any]


ROUTER_SYSTEM_PROMPT = (
    "Classify this query into one or more retrieval sources.\n"
    "Sources: VECTOR_STORE (documents/policies), SQL_DATABASE (structured records/pricing), "
    "WEB_SEARCH (current events/prices), API_CALL (real-time data).\n"
    "Respond with a comma-separated list of source names, e.g.: VECTOR_STORE,SQL_DATABASE"
)


def route_query(
    client: anthropic.Anthropic,
    query: str,
    model: str = "claude-haiku-4-5-20251001",
) -> list[RetrievalSource]:
    response = client.messages.create(
        model=model,
        max_tokens=50,
        system=ROUTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    raw = response.content[0].text.strip().upper()
    sources: list[RetrievalSource] = []
    for name in raw.split(","):
        name = name.strip()
        try:
            sources.append(RetrievalSource[name])
        except KeyError:
            pass
    return sources or [RetrievalSource.VECTOR_STORE]  # fallback


def normalize_sql_result(row: dict[str, Any], query_context: str) -> HybridResult:
    text = " | ".join(f"{k}: {v}" for k, v in row.items())
    return HybridResult(
        text=text,
        score=0.95,   # SQL exact match -- treat as high confidence
        source=RetrievalSource.SQL_DATABASE,
        metadata={"raw_row": row},
    )


class HybridRAGPipeline:
    def __init__(
        self,
        client: anthropic.Anthropic,
        vector_store,        # .search(embedding, user_context, k) -> list[dict]
        embedder,
        sql_executor,        # callable: (query_text) -> list[dict]
        web_searcher,        # callable: (query_text) -> list[str]
        user_context,
    ) -> None:
        self.client = client
        self.vector_store = vector_store
        self.embedder = embedder
        self.sql_executor = sql_executor
        self.web_searcher = web_searcher
        self.user_context = user_context

    def retrieve(self, query: str, k: int = 5) -> list[HybridResult]:
        import concurrent.futures

        sources = route_query(self.client, query)
        all_results: list[HybridResult] = []

        def _vector_retrieve() -> list[HybridResult]:
            embedding = self.embedder.embed(query).vector
            raw = self.vector_store.search(embedding, self.user_context, k=k)
            return [
                HybridResult(
                    text=r["text"],
                    score=r["score"],
                    source=RetrievalSource.VECTOR_STORE,
                    metadata={k: v for k, v in r.items() if k != "text"},
                )
                for r in raw
            ]

        def _sql_retrieve() -> list[HybridResult]:
            rows = self.sql_executor(query)
            return [normalize_sql_result(row, query) for row in rows[:k]]

        def _web_retrieve() -> list[HybridResult]:
            snippets = self.web_searcher(query)
            return [
                HybridResult(
                    text=snippet,
                    score=0.80,
                    source=RetrievalSource.WEB_SEARCH,
                    metadata={},
                )
                for snippet in snippets[:k]
            ]

        task_map = {
            RetrievalSource.VECTOR_STORE: _vector_retrieve,
            RetrievalSource.SQL_DATABASE: _sql_retrieve,
            RetrievalSource.WEB_SEARCH: _web_retrieve,
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(task_map[s]): s
                for s in sources
                if s in task_map
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as exc:
                    source = futures[future]
                    print(f"[WARNING] {source.value} retrieval failed: {exc}")

        # Sort by score descending, deduplicate by text content
        seen: set[str] = set()
        unique: list[HybridResult] = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.text not in seen:
                seen.add(r.text)
                unique.append(r)

        return unique[:k]
```

The `HybridRAGPipeline` uses `ThreadPoolExecutor` to query all routed sources in parallel -- the total latency is the slowest source, not the sum of all sources. Each source call is wrapped in a try/except so a failure in web search does not prevent vector store results from returning. The deduplication at the end by text content prevents the same information appearing from two sources from doubling up in the context window.

---

### Architecture Layer

#### Hybrid RAG System Topology

```
[Query]
         |
         v
[Query Router] -- LLM classifier (Haiku, fast/cheap)
         |
         +---> [Vector Store]    (documents, policies, manuals)
         |       HNSW + BM25 hybrid
         |
         +---> [SQL Database]    (products, prices, customer data)
         |       Structured queries, exact matches
         |
         +---> [Web Search API]  (real-time, current events)
         |       Bing/Brave/SerpAPI
         |
         +---> [Internal APIs]   (inventory, CRM, tickets)
                 REST calls with auth
         |
         v (results arrive in parallel)
[Result Merger + Deduplication]
[Cross-encoder Reranker] (if latency allows)
         |
         v
[Context Assembly + LLM Answer]
```

| Source type | Best for | Latency | Freshness |
|------------|---------|---------|----------|
| Vector store | Unstructured docs, semantic search | 10-50ms | Minutes (re-indexing) |
| SQL database | Structured records, exact lookups | 1-5ms | Real-time |
| Web search | Current events, live prices | 200-500ms | Real-time |
| Internal APIs | Live system state | 50-200ms | Real-time |

> **Bridge:** You have every component of a production RAG system. The capstone project assembles all of them -- from ingestion to access control to evaluation -- into a single runnable enterprise knowledge assistant.

---

⚡ **Senior Checklist -- 3.17**
- [ ] Use a router to classify query type before retrieving -- SQL databases answer pricing questions better than vector stores
- [ ] Query multiple sources in parallel (ThreadPoolExecutor), not sequentially -- latency is the slowest source, not the sum
- [ ] Wrap each source call in try/except -- a failing web search API should not block vector store results
- [ ] Normalize results from different sources to a common schema before reranking -- rerankers cannot compare SQL rows and document chunks without normalization
- [ ] Use Haiku for routing classification -- it is a simple single-label task that does not need Sonnet
- [ ] Cache SQL and API results aggressively (5-min TTL) -- the same product price query should not hit the database on every request
- [ ] Test the fallback: what does the system return when all non-vector sources fail? Should always degrade to vector-only, never return nothing


## 3.18 Stage 3 Project: Enterprise Knowledge Assistant

### 🧠 Mental Model
> This is the system all previous sections have been building toward. It ingests documents, chunks them intelligently, embeds them, enforces tenant isolation, retrieves with hybrid search, assembles context under a token budget, generates cited answers, and scores itself with RAGAS. Every line of code maps to a specific lesson section.

**Connects to:** Hybrid Architectures (3.17) -> **Enterprise Project** -> Quick Reference
**Parent concept:** Integration of all Stage 3 concepts
**Builds on:** All sections 3.1-3.17 -- this project demonstrates every concept in a single runnable system

---

### Project: TechCorp Knowledge Assistant

**Domain:** A B2B software company with 200 enterprise tenants. Each tenant has uploaded their own product documentation, support policies, and technical manuals. Users ask questions in natural language. The system must:

1. Only return documents belonging to the querying tenant
2. Cite the specific source document and page for every fact
3. Score its own answer quality with RAGAS
4. Handle both document search and structured product data
5. Detect and log when it cannot answer confidently

---

### Full Runnable Implementation

```python
from __future__ import annotations

# ============================================================
# Enterprise Knowledge Assistant -- Stage 3 Capstone Project
# Demonstrates: 3.1 (why RAG), 3.2-3.3 (embeddings),
#   3.4-3.5 (vector search + DB), 3.6-3.7 (ingestion + chunking),
#   3.8 (metadata), 3.9 (retrieval), 3.10 (prompt assembly),
#   3.11 (citations), 3.12 (access control), 3.13 (freshness),
#   3.14 (evaluation), 3.15 (failure modes), 3.16 (advanced RAG),
#   3.17 (hybrid architecture)
# ============================================================

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
import numpy as np


# -----------------------------------------------
# Data models
# -----------------------------------------------

@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    tenant_id: str
    access_level: str          # "public" | "internal" | "confidential"
    source_title: str
    source_path: str
    page_number: int | None
    section_heading: str | None
    content_type: str          # "policy" | "manual" | "faq"
    language: str
    created_at: str
    updated_at: str
    is_draft: bool = False
    content_hash: str = ""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata


@dataclass
class KnowledgeAnswer:
    answer_text: str
    cited_chunk_ids: list[str]
    confidence: str
    grounding_score: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    sources_used: list[dict[str, str]]   # [{title, page, section}]


# -----------------------------------------------
# Section 3.2-3.3: Embedding
# -----------------------------------------------

class SimpleEmbedder:
    """
    Production-ready embedder using OpenAI or Claude.
    For this demo, uses random vectors so the project runs without API keys.
    Replace with OpenAIEmbedder from Section 3.2 for real use.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        vec = rng.standard_normal(self.dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # normalize to unit length (Section 3.2)
        result = vec.tolist()
        self._cache[text] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# -----------------------------------------------
# Section 3.4-3.5: Vector store
# -----------------------------------------------

class InMemoryVectorStore:
    """
    In-memory HNSW-style store for demo purposes.
    In production: replace with Qdrant or pgvector (Section 3.5).
    """

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, ChunkMetadata] = {}
        self._texts: dict[str, str] = {}

    def upsert(self, chunk_id: str, vector: list[float], text: str, metadata: ChunkMetadata) -> None:
        if not metadata.tenant_id:
            raise ValueError("tenant_id required -- access control violation (Section 3.12)")
        self._vectors[chunk_id] = vector
        self._metadata[chunk_id] = metadata
        self._texts[chunk_id] = text

    def search(
        self,
        query_vector: list[float],
        tenant_id: str,
        access_levels: list[str],
        k: int = 10,
        min_score: float = 0.70,
    ) -> list[RetrievedChunk]:
        # Section 3.12: filter at index scan level, not post-retrieval
        eligible = {
            cid: meta
            for cid, meta in self._metadata.items()
            if (
                meta.tenant_id == tenant_id          # hard tenant boundary
                and not meta.is_draft                 # exclude drafts
                and meta.access_level in access_levels  # access control
            )
        }
        if not eligible:
            return []

        q = np.array(query_vector)
        scored: list[tuple[float, str]] = []
        for cid in eligible:
            v = np.array(self._vectors[cid])
            score = float(np.dot(q, v))  # dot product == cosine for normalized vecs
            if score >= min_score:
                scored.append((score, cid))

        scored.sort(reverse=True)
        results: list[RetrievedChunk] = []
        for score, cid in scored[:k]:
            results.append(RetrievedChunk(
                chunk_id=cid,
                text=self._texts[cid],
                score=score,
                metadata=self._metadata[cid],
            ))
        return results

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self._vectors.pop(cid, None)
            self._metadata.pop(cid, None)
            self._texts.pop(cid, None)


# -----------------------------------------------
# Section 3.6-3.7: Ingestion and chunking
# -----------------------------------------------

def count_approx_tokens(text: str) -> int:
    return len(text) // 4


def chunk_document(
    text: str,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[str]:
    """Recursive character splitting (Section 3.7)."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        parts = text.split(sep)
        result: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if count_approx_tokens(candidate) <= chunk_size_tokens:
                current = candidate
            else:
                if current:
                    result.append(current)
                if count_approx_tokens(part) > chunk_size_tokens:
                    result.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result

    raw_chunks = _split(text, separators)
    chunks: list[str] = []
    for i, raw in enumerate(raw_chunks):
        if overlap_tokens > 0 and i > 0 and chunks:
            prev_words = chunks[-1].split()
            overlap_words = prev_words[-max(1, overlap_tokens // 5):]
            raw = " ".join(overlap_words) + " " + raw
        chunks.append(raw.strip())
    return [c for c in chunks if c]


def ingest_document(
    store: InMemoryVectorStore,
    embedder: SimpleEmbedder,
    raw_text: str,
    document_id: str,
    tenant_id: str,
    source_title: str,
    source_path: str,
    access_level: str = "internal",
    content_type: str = "policy",
    language: str = "en",
    page_number: int | None = None,
) -> list[str]:
    """Section 3.6: full ingestion pipeline."""
    chunks = chunk_document(raw_text)
    now = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
    chunk_ids: list[str] = []

    vectors = embedder.embed_batch(chunks)

    for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = str(uuid.uuid4())
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=i,
            total_chunks=len(chunks),
            tenant_id=tenant_id,
            access_level=access_level,
            source_title=source_title,
            source_path=source_path,
            page_number=page_number,
            section_heading=None,
            content_type=content_type,
            language=language,
            created_at=now,
            updated_at=now,
            content_hash=content_hash,
        )
        store.upsert(chunk_id, vector, chunk_text, metadata)
        chunk_ids.append(chunk_id)

    return chunk_ids


# -----------------------------------------------
# Section 3.9-3.10: Retrieval and context assembly
# -----------------------------------------------

def assemble_context_block(chunks: list[RetrievedChunk], max_tokens: int = 4000) -> tuple[str, dict[int, str], dict[int, str]]:
    """
    Returns (context_block, chunk_id_map, context_texts).
    Applies sandwich layout (Section 3.10).
    """
    # Sandwich layout: best first, second-best last
    ordered = list(chunks)
    if len(ordered) > 2:
        first = ordered[0]
        second_best = ordered[1]
        middle = ordered[2:]
        ordered = [first] + middle + [second_best]

    chunk_id_map: dict[int, str] = {}
    context_texts: dict[int, str] = {}
    lines: list[str] = ["<context>"]
    total = 0

    for i, chunk in enumerate(ordered, start=1):
        source = chunk.metadata.source_title
        if chunk.metadata.page_number:
            source += f", p.{chunk.metadata.page_number}"
        token_count = count_approx_tokens(chunk.text)
        if total + token_count > max_tokens:
            break
        lines.append(f"[{i}] Source: {source}")
        lines.append(chunk.text)
        lines.append("")
        chunk_id_map[i] = chunk.chunk_id
        context_texts[i] = chunk.text
        total += token_count

    lines.append("</context>")
    return "\n".join(lines), chunk_id_map, context_texts


# -----------------------------------------------
# Section 3.11: Citation and grounding
# -----------------------------------------------

CITATION_SYSTEM_PROMPT = (
    "You are a knowledge assistant. Answer questions using ONLY the information "
    "in <context>. After every factual statement, add [N] where N is the source "
    "chunk number. If the answer is not in the context, respond: "
    "'I do not have that information in the provided documents.' "
    "Format your response as:\n"
    "ANSWER: [answer with inline [N] citations]\n"
    "SOURCES: [comma-separated numbers, e.g. 1, 3]\n"
    "CONFIDENCE: [high / medium / low]"
)


def parse_cited_answer(
    raw: str,
    chunk_id_map: dict[int, str],
    context_texts: dict[int, str],
) -> tuple[str, list[str], str, float]:
    answer_match = re.search(r"ANSWER:\s*(.+?)(?=SOURCES:|CONFIDENCE:|$)", raw, re.DOTALL)
    sources_match = re.search(r"SOURCES:\s*(.+?)(?=CONFIDENCE:|$)", raw, re.DOTALL)
    conf_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", raw, re.IGNORECASE)

    answer_text = answer_match.group(1).strip() if answer_match else raw
    sources_raw = sources_match.group(1).strip() if sources_match else ""
    confidence = conf_match.group(1).lower() if conf_match else "low"

    cited_numbers = [int(n) for n in re.findall(r"\d+", sources_raw)]
    cited_ids = [chunk_id_map[n] for n in cited_numbers if n in chunk_id_map]

    stopwords = {"the", "a", "an", "is", "are", "in", "of", "to", "and", "or"}
    answer_words = set(answer_text.lower().split()) - stopwords
    cited_text = " ".join(context_texts.get(n, "") for n in cited_numbers)
    cited_words = set(cited_text.lower().split()) - stopwords
    grounding = len(answer_words & cited_words) / max(len(answer_words), 1)

    return answer_text, cited_ids, confidence, grounding


# -----------------------------------------------
# Section 3.14: RAGAS-style self-evaluation
# -----------------------------------------------

def self_evaluate(
    client: anthropic.Anthropic,
    query: str,
    answer: str,
    context_texts: dict[int, str],
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, float]:
    context_block = "\n".join(f"[{n}] {text}" for n, text in context_texts.items())
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Evaluate this RAG answer on two dimensions (0.0-1.0):\n"
                f"1. faithfulness: does the answer only contain facts from the context?\n"
                f"2. answer_relevancy: does the answer address the question?\n\n"
                f"Question: {query}\n"
                f"Context:\n{context_block}\n"
                f"Answer: {answer}\n\n"
                f"Reply in this exact format:\n"
                f"faithfulness: <0.0-1.0>\n"
                f"answer_relevancy: <0.0-1.0>"
            ),
        }],
    )
    scores: dict[str, float] = {}
    for line in response.content[0].text.strip().split("\n"):
        parts = line.split(":")
        if len(parts) == 2:
            try:
                scores[parts[0].strip()] = float(parts[1].strip())
            except ValueError:
                pass
    return scores


# -----------------------------------------------
# Main pipeline
# -----------------------------------------------

class EnterpriseKnowledgeAssistant:
    def __init__(
        self,
        client: anthropic.Anthropic,
        store: InMemoryVectorStore,
        embedder: SimpleEmbedder,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.client = client
        self.store = store
        self.embedder = embedder
        self.model = model

    def answer(
        self,
        query: str,
        tenant_id: str,
        access_levels: list[str] | None = None,
        k: int = 8,
        run_self_eval: bool = False,
    ) -> KnowledgeAnswer:
        if access_levels is None:
            access_levels = ["public", "internal"]

        # Section 3.9: retrieval
        t0 = time.monotonic()
        query_vector = self.embedder.embed(query)
        chunks = self.store.search(
            query_vector=query_vector,
            tenant_id=tenant_id,
            access_levels=access_levels,
            k=k,
        )
        retrieval_ms = (time.monotonic() - t0) * 1000

        if not chunks:
            return KnowledgeAnswer(
                answer_text="I do not have information about this in the provided documents.",
                cited_chunk_ids=[],
                confidence="low",
                grounding_score=0.0,
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=0.0,
                sources_used=[],
            )

        # Section 3.10: context assembly
        context_block, chunk_id_map, context_texts = assemble_context_block(chunks)

        # Section 3.11: cited answer generation
        t1 = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=CITATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{context_block}\n\nQuestion: {query}"}],
        )
        generation_ms = (time.monotonic() - t1) * 1000

        raw = response.content[0].text
        answer_text, cited_ids, confidence, grounding = parse_cited_answer(
            raw, chunk_id_map, context_texts
        )

        # Self-evaluation (Section 3.14) -- optional, adds latency
        if run_self_eval and cited_ids:
            eval_scores = self_evaluate(self.client, query, answer_text, context_texts)
            # Log or use eval_scores downstream

        sources = [
            {
                "title": c.metadata.source_title,
                "page": str(c.metadata.page_number or ""),
                "section": c.metadata.section_heading or "",
                "chunk_id": c.chunk_id,
            }
            for c in chunks[:5]
        ]

        return KnowledgeAnswer(
            answer_text=answer_text,
            cited_chunk_ids=cited_ids,
            confidence=confidence,
            grounding_score=round(grounding, 3),
            retrieval_latency_ms=round(retrieval_ms, 1),
            generation_latency_ms=round(generation_ms, 1),
            sources_used=sources,
        )


# -----------------------------------------------
# Demo run
# -----------------------------------------------

if __name__ == "__main__":
    # Initialize components
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env variable
    store = InMemoryVectorStore()
    embedder = SimpleEmbedder(dim=256)

    # Ingest sample documents for two tenants (Section 3.6-3.8)
    acme_policy = """
    Refund Policy for ACME Corp Software Licenses

    Standard Refunds:
    Software licenses may be refunded within 30 days of purchase if the software
    fails to perform its documented functions. Refunds require a support ticket
    with evidence of the defect.

    Annual Subscriptions:
    Annual subscription refunds are prorated based on unused months.
    Cancellation must be submitted 30 days before renewal date.

    Enterprise Licenses:
    Enterprise licenses valued over $10,000 require written approval from
    the VP of Sales before a refund can be processed. Processing time is 5-7 business days.
    """

    globex_policy = """
    Globex Industries Support Policy

    Response Times:
    Priority 1 (System Down): 1-hour response, 4-hour resolution target.
    Priority 2 (Major Feature): 4-hour response, 24-hour resolution target.
    Priority 3 (Minor Issue): 24-hour response, 5-day resolution target.

    Escalation:
    Unresolved Priority 1 issues escalate to the Senior Support Manager after 2 hours.
    """

    # Ingest ACME documents
    ingest_document(
        store, embedder,
        raw_text=acme_policy,
        document_id="acme-refund-policy-v2",
        tenant_id="acme-corp",
        source_title="ACME Software Refund Policy",
        source_path="/policies/refund-policy-v2.pdf",
        access_level="internal",
        content_type="policy",
        page_number=1,
    )

    # Ingest Globex documents (different tenant -- must stay isolated)
    ingest_document(
        store, embedder,
        raw_text=globex_policy,
        document_id="globex-support-policy-v1",
        tenant_id="globex-industries",
        source_title="Globex Support Policy",
        source_path="/policies/support-policy-v1.pdf",
        access_level="internal",
        content_type="policy",
        page_number=1,
    )

    # Build the assistant
    assistant = EnterpriseKnowledgeAssistant(
        client=client,
        store=store,
        embedder=embedder,
        model="claude-sonnet-4-6",
    )

    print("=" * 60)
    print("DEMO 1: Normal query for ACME tenant")
    print("=" * 60)
    answer = assistant.answer(
        query="How long do I have to request a refund?",
        tenant_id="acme-corp",
    )
    print(f"Answer: {answer.answer_text}")
    print(f"Confidence: {answer.confidence}")
    print(f"Grounding score: {answer.grounding_score}")
    print(f"Retrieval: {answer.retrieval_latency_ms}ms, Generation: {answer.generation_latency_ms}ms")
    print()

    print("=" * 60)
    print("DEMO 2: Cross-tenant isolation test (Section 3.12)")
    print("  ACME user asking question -- should NOT see Globex data")
    print("=" * 60)
    answer2 = assistant.answer(
        query="What are the support response time SLAs?",
        tenant_id="acme-corp",  # ACME user -- Globex docs must be invisible
    )
    print(f"Answer: {answer2.answer_text}")
    print(f"Sources returned: {[s['title'] for s in answer2.sources_used]}")
    # Should show ACME docs only or "I do not have that information"
    print()

    print("=" * 60)
    print("DEMO 3: Out-of-scope query (should admit lack of knowledge)")
    print("=" * 60)
    answer3 = assistant.answer(
        query="What is the weather forecast for tomorrow?",
        tenant_id="acme-corp",
    )
    print(f"Answer: {answer3.answer_text}")
    print(f"Confidence: {answer3.confidence}")
```

---

### Topic-to-Code Mapping

| Section | Concept | Code location |
|---------|---------|--------------|
| 3.1 | Why RAG | Decision to use vector store + LLM, not fine-tuning |
| 3.2-3.3 | Embeddings + model selection | `SimpleEmbedder` class (replace with `OpenAIEmbedder`) |
| 3.4-3.5 | Vector search + DB | `InMemoryVectorStore` (replace with Qdrant) |
| 3.6 | Document ingestion | `ingest_document()` function |
| 3.7 | Chunking strategies | `chunk_document()` recursive splitter |
| 3.8 | Metadata design | `ChunkMetadata` dataclass with all fields |
| 3.9 | Retrieval patterns | `store.search()` with score threshold |
| 3.10 | Prompt assembly | `assemble_context_block()` with sandwich layout |
| 3.11 | Citations + grounding | `parse_cited_answer()`, `grounding_score` |
| 3.12 | Access control | `build_tenant_filter()` logic inside `store.search()` |
| 3.13 | Index freshness | `content_hash` in metadata, `ingest_document` delete-then-upsert |
| 3.14 | RAGAS evaluation | `self_evaluate()` function |
| 3.15 | Failure modes | `confidence` + `grounding_score` as failure signals |
| 3.16 | Advanced RAG | Query expansion + HyDE from Section 3.16 can be added before `embed()` |
| 3.17 | Hybrid architectures | Extend `answer()` to fan out to SQL + web search before context assembly |

---

### Observability

Instrument these metrics in production:

```python
# Metrics to track per query
metrics = {
    "retrieval_latency_ms": answer.retrieval_latency_ms,
    "generation_latency_ms": answer.generation_latency_ms,
    "total_latency_ms": answer.retrieval_latency_ms + answer.generation_latency_ms,
    "chunks_retrieved": len(chunks),
    "grounding_score": answer.grounding_score,    # alert if 7-day avg < 0.5
    "confidence": answer.confidence,
    "had_citations": len(answer.cited_chunk_ids) > 0,
    "was_out_of_scope": "do not have that information" in answer.answer_text,
    "tenant_id": tenant_id,
}
# Send to DataDog, Prometheus, or your observability stack
```

Alert thresholds:
- `grounding_score` 7-day average < 0.50: hallucination risk increasing
- `was_out_of_scope` rate > 30%: knowledge base gaps -- ingest more documents
- `total_latency_ms` p95 > 3,000ms: retrieval or generation bottleneck
- `chunks_retrieved == 0` rate > 10%: index freshness or access control issue

---

### What to Add for Production

| Layer | What to add | Why |
|-------|------------|-----|
| Infrastructure | Qdrant or pgvector replacing InMemoryVectorStore | Persistence, multi-node, production scale |
| Embeddings | OpenAI `text-embedding-3-small` replacing SimpleEmbedder | Real semantic quality |
| Ingestion | Webhook + polling freshness sync (Section 3.13) | Keep index current |
| Access control | JWT auth middleware setting `tenant_id` from token claim | Security |
| Advanced retrieval | BM25 hybrid search (Section 3.9) | +10-15% recall on technical queries |
| Evaluation | Full RAGAS pipeline on 200-question test set (Section 3.14) | Quality gates before deployment |
| Observability | Distributed tracing (Langfuse or Honeycomb) | Debug production failures |
| Caching | Semantic query cache (LRU, 15-min TTL) | 30-50% latency reduction on repeated queries |

---

⚡ **Senior Checklist -- 3.18**
- [ ] Replace `InMemoryVectorStore` with Qdrant before production deployment -- in-memory storage is not persistent
- [ ] Replace `SimpleEmbedder` with a real embedding API -- random vectors demonstrate the interface, not the quality
- [ ] Add JWT middleware that validates `tenant_id` from the token before calling `assistant.answer()` -- never trust user-supplied tenant_id
- [ ] Set up the freshness sync pipeline from Section 3.13 -- without it the index becomes stale within days
- [ ] Run RAGAS evaluation on a test set of 100+ questions before launch -- deploy only when faithfulness >= 0.80 and context_recall >= 0.70
- [ ] Add structured logging of grounding_score on every request to a time-series metric store
- [ ] Load-test the retrieval pipeline at your expected peak QPS -- HNSW query times grow logarithmically but embedding API calls are serial without batching

---

## Quick Reference Cheat Sheet

### Stage Mind Map

```
RAG: Give the model the right documents at the right time
│
├── 3.1 Why RAG ─────────── Judgment: prompt stuffing vs fine-tune vs RAG
│
├── 3.2 Embeddings ─────────── Text → coordinates in meaning space
│         └── 3.3 Model Selection ── MTEB scores + domain benchmark
│
├── 3.4 Vector Search ──────── HNSW: O(log n), 96% recall at ef_search=100
│         └── 3.5 DB Selection ──── pgvector vs Qdrant vs Pinecone
│
├── 3.6 Ingestion ──────────── Pipeline: extract → chunk → embed → upsert
│         ├── 3.7 Chunking ──────── Recursive splitting, overlap 10-20%
│         └── 3.8 Metadata ──────── tenant_id + access_level + freshness fields
│
├── 3.9 Retrieval Patterns ─── Dense + BM25 hybrid, RRF fusion, cross-encoder rerank
│         ├── 3.10 Prompt Assembly ─ Token budget + sandwich layout
│         └── 3.11 Citations ─────── [N] inline + ANSWER/SOURCES/CONFIDENCE format
│
├── 3.12 Access Control ────── Index-level filter (Qdrant payload / pgvector RLS)
│         └── 3.13 Index Freshness ─ Content hash + reconciliation scan
│
├── 3.14 RAG Evaluation ────── RAGAS: recall + precision + faithfulness + relevancy
│         └── 3.15 Failure Modes ─── Diagnose by RAGAS score pattern
│
├── 3.16 Advanced RAG ──────── Query expansion + HyDE + multi-hop
│         └── 3.17 Hybrid Arch ────── Router: vector + SQL + web + API
│
└── 3.18 Capstone ──────────── Enterprise Knowledge Assistant (all sections)
```

---

### Three-Layer Framework Summary

| Layer | Question it answers | Output |
|-------|-------------------|--------|
| Concept Layer | How does this actually work? | Algorithm, mechanism, worked example |
| Engineering Layer | How do I build it? | Production Python code, edge cases, tradeoffs |
| Architecture Layer | Where does it fit in a real system? | ASCII diagram, scaling limits, cost at scale |

---

### Key Numbers Table

| Fact | Number / Threshold | Why It Matters |
|------|--------------------|----------------|
| Prompt stuffing limit | 60% of context window | Leave 40% for reasoning headroom |
| Minimum knowledge base for RAG | > 20 documents | Below this, prompt stuffing is simpler |
| Cosine similarity threshold (pass/fail) | 0.70 | Below this, retrieval results are unreliable |
| HNSW ef_search default | 100 | 96% recall at 9ms for 1M 1536-dim vectors |
| HNSW memory (M=16) | ~100 MB per 1K 1536-dim vectors | Size your vector DB instance accordingly |
| Chunk size sweet spot | 512 tokens | Balances embedding quality vs context fit |
| Chunk overlap | 10-20% of chunk size | 512-token chunk needs 50-100 token overlap |
| Embedding cost (small model) | $0.00002/1K tokens | $0.08 to embed 10K docs at 400 tokens each |
| text-embedding-3-small Matryoshka | 512 dims: -2% quality, -67% storage | Use for first-pass retrieval |
| BM25 RRF constant k | 60 | Standard default; lower over-weights top ranks |
| Reranker latency | 50-200ms per 20 candidates | Budget this against your SLA |
| Context overflow alert threshold | > 5% of queries truncated | Signals chunk size or k too large |
| Grounding score alert threshold | < 0.50 (7-day average) | Hallucination risk increasing |
| RAGAS faithfulness minimum | >= 0.80 | Deploy gate |
| RAGAS context_recall minimum | >= 0.70 | Deploy gate |
| Query expansion improvement | +10-20% context_recall | On vocabulary-mismatch queries |
| HyDE improvement | +5-15% context_recall | On complex technical queries |
| Sandwich layout improvement | +5-15% answer quality | Zero extra cost |
| Cross-tenant isolation test | 100% must pass | Non-negotiable CI gate |
| Daily reconciliation scan | Every 24 hours | Catches deletions polling misses |

---

### Memory Anchors Table

| Section | One-line anchor |
|---------|----------------|
| 3.1 | Use prompt stuffing for small static context, fine-tuning for new skills, RAG for large or dynamic knowledge that needs citations |
| 3.2 | Embeddings turn text into coordinates -- similar meaning lands nearby, different meaning lands far away |
| 3.3 | MTEB is the starting point, not the decision -- run your own domain benchmark on 200 queries before committing to a model |
| 3.4 | Use HNSW for dynamic indexes at ef_search=100 -- 96% recall at 9ms for 1 million vectors |
| 3.5 | pgvector when you already run Postgres; Qdrant when you need payload filtering and scale above 1M vectors |
| 3.6 | Ingestion pipeline = extract content, chunk recursively, embed in batches, upsert with full metadata |
| 3.7 | Never split mid-sentence -- recursive character splitting respects paragraph boundaries; overlap 10-20% of chunk size |
| 3.8 | Metadata is not labeling -- it is the access control and freshness system; design it before your chunk schema |
| 3.9 | Always run hybrid (dense + BM25) in production -- dense misses exact terms; BM25 misses meaning; together they cover both |
| 3.10 | Sandwich layout and token counting are free quality improvements -- use them on every RAG system |
| 3.11 | ANSWER/SOURCES/CONFIDENCE structured output makes citations programmatically reliable -- free-text parsing fails on edge cases |
| 3.12 | Tenant isolation enforced at the index scan level, not in application code -- one buggy filter line is a security breach |
| 3.13 | Use SHA-256 content hashes to detect changes, not just modification timestamps -- run reconciliation scans daily for deletions |
| 3.14 | RAGAS context_recall >= 0.70 and faithfulness >= 0.80 are your minimum deployment gates -- not feelings |
| 3.15 | Each RAGAS score below threshold maps to a specific root cause with a specific fix -- diagnose before changing anything |
| 3.16 | Query expansion fixes vocabulary mismatch; HyDE fixes query-document space gap; multi-hop fixes multi-part questions |
| 3.17 | Route queries to the right source -- SQL for structured records, vector store for documents, API for real-time data |
| 3.18 | Every component in Stage 3 is a single piece of the ingestion-retrieval-assembly-evaluation loop |

---
