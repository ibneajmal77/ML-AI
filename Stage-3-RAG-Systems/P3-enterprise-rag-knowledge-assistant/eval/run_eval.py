"""RAGAS evaluation runner. Exit code 1 if quality thresholds are not met."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from eval.ragas_config import (
    ANSWER_RELEVANCE_THRESHOLD,
    CONTEXT_PRECISION_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
)


async def _run_query(query: str, user_clearance: int = 1) -> dict:
    from src.api.deps import get_hybrid_retriever, get_query_cache, get_vector_store
    from src.context.assembler import assemble_context
    from src.domain.models import AccessLevel
    from src.embedding.encoder import get_query_embedding
    from src.generation.generator import generate_answer
    from src.retrieval.reranker import rerank
    from src.config import settings

    retriever = await get_hybrid_retriever()
    vs = await get_vector_store()
    cache = await get_query_cache()

    cached = await cache.get(query, user_clearance, None)
    if cached:
        return cached

    query_embedding = await get_query_embedding(query)
    candidates = await retriever.search(query, query_embedding, user_clearance, None, settings.reranker_top_k)

    if not candidates:
        return {"answer": "I could not find relevant information.", "contexts": []}

    reranked = await rerank(query, candidates, top_n=settings.final_top_k)
    context, citations = await assemble_context(query, reranked, vs)
    answer = await generate_answer(query, context, citations)

    contexts = [c.chunk.content for c in reranked]
    return {"answer": answer, "contexts": contexts}


def main() -> None:
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]
        user_clearance = item.get("access_level", 1)

        result = asyncio.run(_run_query(question, user_clearance=user_clearance))

        questions.append(question)
        answers.append(result.get("answer", ""))
        contexts.append(result.get("contexts", []))
        ground_truths.append(ground_truth)

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(results)

    failed = False
    if results["faithfulness"] < FAITHFULNESS_THRESHOLD:
        print(f"FAIL: faithfulness {results['faithfulness']:.3f} < {FAITHFULNESS_THRESHOLD}")
        failed = True
    if results["answer_relevancy"] < ANSWER_RELEVANCE_THRESHOLD:
        print(f"FAIL: answer_relevancy {results['answer_relevancy']:.3f} < {ANSWER_RELEVANCE_THRESHOLD}")
        failed = True

    if failed:
        sys.exit(1)

    print("All quality thresholds passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
