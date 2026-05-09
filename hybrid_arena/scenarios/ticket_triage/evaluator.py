"""Evaluation for ticket triage."""

from __future__ import annotations

from hybrid_arena.core.schema import BenchmarkResult
from hybrid_arena.scenarios.ticket_triage.classifier import classify_ticket


def evaluate_ticket_cases(cases: list[dict]) -> BenchmarkResult:
    case_results: list[dict] = []
    expected_labels = sorted({case["expected_label"] for case in cases})
    correct = 0
    unknown_count = 0

    for case in cases:
        prediction = classify_ticket(case["ticket_text"])
        is_correct = prediction.label == case["expected_label"]
        correct += int(is_correct)
        unknown_count += int(prediction.label == "unknown")
        case_results.append(
            {
                "case_id": case.get("case_id", ""),
                "expected": case["expected_label"],
                "actual": prediction.label,
                "correct": is_correct,
            }
        )

    total = len(cases)
    return BenchmarkResult(
        scenario="ticket_triage",
        total=total,
        metrics={
            "accuracy": correct / total if total else 0.0,
            "macro_f1": _macro_f1(case_results, expected_labels),
            "unknown_rate": unknown_count / total if total else 0.0,
        },
        cases=case_results,
    )


def _macro_f1(case_results: list[dict], labels: list[str]) -> float:
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for case in case_results if case["expected"] == label and case["actual"] == label)
        fp = sum(1 for case in case_results if case["expected"] != label and case["actual"] == label)
        fn = sum(1 for case in case_results if case["expected"] == label and case["actual"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)
