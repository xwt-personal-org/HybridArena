from __future__ import annotations

from hybrid_arena.scenarios.telecom_rag.evaluator import evaluate_rag_cases


def test_evaluate_rag_cases_reports_recall_and_citation_coverage() -> None:
    cases = [{"question": "AMF 的职责是什么", "gold_doc_id": "5gc-amf"}]

    result = evaluate_rag_cases(cases, top_k=2)

    assert result.total == 1
    assert result.metrics["recall_at_k"] == 1.0
    assert result.metrics["citation_coverage"] == 1.0
