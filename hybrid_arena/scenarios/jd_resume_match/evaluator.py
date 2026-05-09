"""Evaluation helpers for JD skill extraction."""

from __future__ import annotations

from hybrid_arena.core.schema import BenchmarkResult
from hybrid_arena.scenarios.jd_resume_match.extractor import extract_jd_requirements


def evaluate_skill_extraction(cases: list[dict]) -> BenchmarkResult:
    case_results: list[dict] = []
    recall_values: list[float] = []
    coverage_values: list[float] = []

    for case in cases:
        expected = set(case.get("expected_skills", []))
        requirements = extract_jd_requirements(case["jd_text"])
        actual = {requirement.skill_id for requirement in requirements}
        hits = expected & actual
        recall = len(hits) / len(expected) if expected else 1.0
        evidence_coverage = (
            sum(1 for requirement in requirements if requirement.evidence) / len(requirements)
            if requirements
            else 1.0
        )
        recall_values.append(recall)
        coverage_values.append(evidence_coverage)
        case_results.append(
            {
                "case_id": case.get("case_id", ""),
                "expected": sorted(expected),
                "actual": sorted(actual),
                "skill_recall": recall,
                "evidence_coverage": evidence_coverage,
            }
        )

    total = len(cases)
    return BenchmarkResult(
        scenario="jd_resume_match",
        total=total,
        metrics={
            "skill_recall": sum(recall_values) / total if total else 0.0,
            "evidence_coverage": sum(coverage_values) / total if total else 0.0,
        },
        cases=case_results,
    )
