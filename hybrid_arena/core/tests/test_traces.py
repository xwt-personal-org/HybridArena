from __future__ import annotations

import json

from hybrid_arena.core.traces import JsonlTraceWriter, TraceRecorder


def test_trace_recorder_records_steps_and_metrics() -> None:
    recorder = TraceRecorder(run_id="run-001", task_id="task-001", scenario="telecom_rag")

    recorder.start_step("retrieve", {"query": "什么是 AMF"})
    recorder.finish_step(
        {"chunks": ["5GC core"]},
        metrics={"top_k": 1},
    )

    trace = recorder.to_trace(metrics={"recall_at_k": 1.0})

    assert trace.run_id == "run-001"
    assert trace.steps[0].name == "retrieve"
    assert trace.steps[0].success is True
    assert trace.metrics["recall_at_k"] == 1.0
    assert trace.steps[0].latency_ms >= 0


def test_jsonl_trace_writer_appends_trace(tmp_path) -> None:
    recorder = TraceRecorder(run_id="run-001", task_id="task-001", scenario="ticket_triage")
    recorder.start_step("classify", {"text": "基站弱覆盖"})
    recorder.finish_step({"label": "radio_access"})

    path = tmp_path / "traces.jsonl"
    writer = JsonlTraceWriter(path)
    writer.write(recorder.to_trace())

    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    assert payload["run_id"] == "run-001"
    assert payload["steps"][0]["name"] == "classify"
