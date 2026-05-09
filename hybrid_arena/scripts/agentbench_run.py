"""Run AgentBench scenario benchmarks from JSONL inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_arena.core.reporting import benchmark_to_markdown
from hybrid_arena.core.schema import BenchmarkResult
from hybrid_arena.scenarios.jd_resume_match.evaluator import evaluate_skill_extraction
from hybrid_arena.scenarios.telecom_rag.evaluator import evaluate_rag_cases
from hybrid_arena.scenarios.ticket_triage.evaluator import evaluate_ticket_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HybridArena AgentBench benchmarks")
    parser.add_argument("--scenario", choices=["jd_resume_match", "telecom_rag", "ticket_triage"], required=True)
    parser.add_argument("--input", required=True, help="JSONL benchmark input path")
    parser.add_argument("--output", required=True, help="JSON report output path")
    parser.add_argument("--top-k", type=int, default=3, help="Retriever top-k for RAG benchmarks")
    args = parser.parse_args(argv)

    cases = _read_jsonl(Path(args.input))
    benchmark = _run_benchmark(args.scenario, cases, top_k=args.top_k)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        benchmark_to_markdown(benchmark),
        encoding="utf-8",
    )
    return 0


def _run_benchmark(scenario: str, cases: list[dict], top_k: int) -> BenchmarkResult:
    if scenario == "jd_resume_match":
        return evaluate_skill_extraction(cases)
    if scenario == "telecom_rag":
        return evaluate_rag_cases(cases, top_k=top_k)
    if scenario == "ticket_triage":
        return evaluate_ticket_cases(cases)
    raise ValueError(f"Unknown scenario: {scenario}")


def _read_jsonl(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
