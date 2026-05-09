from __future__ import annotations

from hybrid_arena.core.reporting import benchmark_to_markdown
from hybrid_arena.core.schema import BenchmarkResult


def test_benchmark_to_markdown_includes_metrics_and_cases() -> None:
    benchmark = BenchmarkResult(
        scenario="ticket_triage",
        total=1,
        metrics={"accuracy": 1.0},
        cases=[{"case_id": "ticket-001", "expected": "billing", "actual": "billing"}],
    )

    markdown = benchmark_to_markdown(benchmark, title="Ticket Report")

    assert "# Ticket Report" in markdown
    assert "accuracy" in markdown
    assert "ticket-001" in markdown
