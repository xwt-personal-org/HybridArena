from __future__ import annotations

from hybrid_arena.core.schema import TaskInput
from hybrid_arena.scenarios.telecom_rag.runner import TelecomRagRunner


def test_telecom_rag_runner_returns_answer_trace_and_metrics() -> None:
    runner = TelecomRagRunner()
    task = TaskInput(
        task_id="rag-001",
        scenario="telecom_rag",
        payload={"question": "AMF 负责什么", "top_k": 2},
    )

    result = runner.run(task)

    assert result.output["citations"][0]["doc_id"] == "5gc-amf"
    assert result.metrics["citation_count"] >= 1
    assert [step.name for step in result.trace.steps] == ["load_corpus", "retrieve", "generate_answer"]
