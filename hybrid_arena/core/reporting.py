"""Report rendering helpers for AgentBench benchmark results."""

from __future__ import annotations

from hybrid_arena.core.schema import BenchmarkResult


def benchmark_to_markdown(benchmark: BenchmarkResult, title: str | None = None) -> str:
    heading = title or f"{benchmark.scenario} Benchmark Report"
    lines = [f"# {heading}", "", f"- scenario: `{benchmark.scenario}`", f"- total: `{benchmark.total}`", ""]
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key, value in benchmark.metrics.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    if not benchmark.cases:
        lines.append("No cases recorded.")
        return "\n".join(lines) + "\n"

    keys = sorted({key for case in benchmark.cases for key in case})
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join("---" for _ in keys) + "|")
    for case in benchmark.cases:
        lines.append("| " + " | ".join(str(case.get(key, "")) for key in keys) + " |")
    return "\n".join(lines) + "\n"
