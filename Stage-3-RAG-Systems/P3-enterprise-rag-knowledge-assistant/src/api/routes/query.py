from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from src.api.deps import get_hybrid_retriever, get_query_cache, get_vector_store
from src.cache.query_cache import QueryCache
from src.config import settings
from src.context.assembler import assemble_context
from src.domain.models import Citation, QueryRequest, QueryResponse
from src.embedding.encoder import get_query_embedding
from src.generation.generator import generate_answer
from src.retrieval.advanced import generate_query_variations, hyde_embedding, merge_results
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import rerank
from src.store.vector_store import VectorStore
from src.tracing.tracer import RAGTrace

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["query"])

_NOT_FOUND_RESPONSE = QueryResponse(
    answer="I could not find relevant information to answer your question.",
    citations=[],
    latency_ms=0.0,
    cache_hit=False,
)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    vector_store: VectorStore = Depends(get_vector_store),
    cache: QueryCache = Depends(get_query_cache),
) -> QueryResponse:
    start = time.perf_counter()
    trace = RAGTrace(request.query, request.user_clearance.value)

    # Cache check
    _tags = request.tags or None
    cached = await cache.get(
        request.query,
        request.user_clearance.value,
        request.owner_dept,
        use_hyde=request.use_hyde,
        use_multi_query=request.use_multi_query,
        tags=_tags,
    )
    if cached:
        cached["cache_hit"] = True
        cached["latency_ms"] = (time.perf_counter() - start) * 1000
        return QueryResponse(**cached)

    # Embedding
    trace.span("embedding", input_data={"query": request.query, "use_hyde": request.use_hyde})
    if request.use_hyde:
        query_embedding = await hyde_embedding(request.query)
    else:
        query_embedding = await get_query_embedding(request.query)
    trace.end_span("embedding")

    # Retrieval
    trace.span("retrieval", input_data={"use_multi_query": request.use_multi_query})
    if request.use_multi_query:
        variations = await generate_query_variations(request.query)
        result_sets = []
        for variation in variations:
            var_embedding = await get_query_embedding(variation)
            results = await retriever.search(
                variation,
                var_embedding,
                request.user_clearance.value,
                request.owner_dept,
                settings.reranker_top_k,
                tags=_tags,
            )
            result_sets.append(results)
        candidates = merge_results(result_sets)
    else:
        candidates = await retriever.search(
            request.query,
            query_embedding,
            request.user_clearance.value,
            request.owner_dept,
            settings.reranker_top_k,
            tags=_tags,
        )
    trace.end_span("retrieval", output={"candidates": len(candidates)})

    if not candidates:
        latency_ms = (time.perf_counter() - start) * 1000
        trace.end(output={"answer": "no candidates"})
        return QueryResponse(
            answer="I could not find relevant information to answer your question.",
            citations=[],
            latency_ms=latency_ms,
            cache_hit=False,
        )

    # Rerank
    trace.span("rerank", input_data={"candidates": len(candidates)})
    reranked = await rerank(request.query, candidates, top_n=settings.final_top_k)
    trace.end_span("rerank", output={"reranked": len(reranked)})

    # Context assembly
    trace.span("assemble")
    context, citations = await assemble_context(
        request.query, reranked, vector_store, request.conversation_history or None
    )
    trace.end_span("assemble")

    # Generation
    trace.span("generate")
    answer = await generate_answer(
        request.query, context, citations, request.conversation_history or None
    )
    trace.end_span("generate", output={"answer_length": len(answer)})

    latency_ms = (time.perf_counter() - start) * 1000
    response = QueryResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
        cache_hit=False,
    )

    # Cache the result
    await cache.set(
        request.query,
        request.user_clearance.value,
        request.owner_dept,
        response.model_dump(mode="json"),
        use_hyde=request.use_hyde,
        use_multi_query=request.use_multi_query,
        tags=_tags,
    )

    trace.end(output={"answer_length": len(answer), "latency_ms": latency_ms})
    return response
