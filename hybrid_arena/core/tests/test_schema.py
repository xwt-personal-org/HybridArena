from __future__ import annotations

from hybrid_arena.core.schema import (
    BenchmarkResult,
    TaskInput,
    TaskRunResult,
    TaskTrace,
    ToolCallRecord,
)


def test_task_run_result_round_trips_with_trace() -> None:
    task = TaskInput(
        task_id="jd-001",
        scenario="jd_resume_match",
        payload={"jd_text": "需要 Python 和 FastAPI 经验"},
        metadata={"source": "unit"},
    )
    tool_call = ToolCallRecord(
        name="extract_skills",
        input={"text": task.payload["jd_text"]},
        output={"skills": ["python_backend", "http_api"]},
        latency_ms=12.5,
        success=True,
    )
    trace = TaskTrace(
        run_id="run-001",
        task_id=task.task_id,
        scenario=task.scenario,
        steps=[tool_call],
        metrics={"step_count": 1},
    )
    result = TaskRunResult(
        run_id="run-001",
        task_id=task.task_id,
        scenario=task.scenario,
        output={"missing_skills": ["rag"]},
        metrics={"missing_skill_count": 1},
        trace=trace,
    )

    restored = TaskRunResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.trace.steps[0].name == "extract_skills"


def test_benchmark_result_keeps_case_metrics() -> None:
    benchmark = BenchmarkResult(
        scenario="ticket_triage",
        total=2,
        metrics={"macro_f1": 0.75},
        cases=[
            {"case_id": "t1", "expected": "radio_access", "actual": "radio_access"},
            {"case_id": "t2", "expected": "core_network", "actual": "unknown"},
        ],
    )

    restored = BenchmarkResult.from_dict(benchmark.to_dict())

    assert restored.metrics["macro_f1"] == 0.75
    assert restored.cases[1]["actual"] == "unknown"
