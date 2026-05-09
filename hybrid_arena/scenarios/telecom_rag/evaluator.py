"""Evaluation for the telecom RAG scenario."""

from __future__ import annotations

from hybrid_arena.core.schema import BenchmarkResult
from hybrid_arena.scenarios.telecom_rag.corpus import load_corpus
from hybrid_arena.scenarios.telecom_rag.generator import generate_answer
from hybrid_arena.scenarios.telecom_rag.retriever import retrieve


def evaluate_rag_cases(cases: list[dict], top_k: int = 3) -> BenchmarkResult:
    chunks = load_corpus()
    case_results: list[dict] = []
    recall_values: list[float] = []
    citation_values: list[float] = []
    unsupported_values: list[float] = []

    for case in cases:
        results = retrieve(case["question"], chunks, top_k=top_k)
        answer = generate_answer(case["question"], results)
        retrieved_ids = [result.chunk.doc_id for result in results]
        gold_doc_id = case.get("gold_doc_id")
        recall = 1.0 if gold_doc_id in retrieved_ids else 0.0
        citation_coverage = 1.0 if answer["citations"] else 0.0
        unsupported = 1.0 if answer["unsupported"] else 0.0
        recall_values.append(recall)
        citation_values.append(citation_coverage)
        unsupported_values.append(unsupported)
        case_results.append(
            {
                "question": case["question"],
                "gold_doc_id": gold_doc_id,
                "retrieved_doc_ids": retrieved_ids,
                "recall_at_k": recall,
                "citation_coverage": citation_coverage,
                "unsupported": bool(unsupported),
            }
        )

    total = len(cases)
    return BenchmarkResult(
        scenario="telecom_rag",
        total=total,
        metrics={
            "recall_at_k": sum(recall_values) / total if total else 0.0,
            "citation_coverage": sum(citation_values) / total if total else 0.0,
            "unsupported_answer_rate": sum(unsupported_values) / total if total else 0.0,
        },
        cases=case_results,
    )
