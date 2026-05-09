from __future__ import annotations

from hybrid_arena.scenarios.jd_resume_match.evaluator import evaluate_skill_extraction


def test_evaluate_skill_extraction_reports_recall_and_coverage() -> None:
    cases = [
        {
            "jd_text": "熟悉 Python、FastAPI 和 RAG。",
            "expected_skills": ["python_backend", "http_api", "rag"],
        }
    ]

    result = evaluate_skill_extraction(cases)

    assert result.total == 1
    assert result.metrics["skill_recall"] == 1.0
    assert result.metrics["evidence_coverage"] == 1.0
