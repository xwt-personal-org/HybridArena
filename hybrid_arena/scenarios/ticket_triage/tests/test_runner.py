from __future__ import annotations

from hybrid_arena.core.schema import TaskInput
from hybrid_arena.scenarios.ticket_triage.runner import TicketTriageRunner


def test_ticket_triage_runner_returns_prediction_steps_and_trace() -> None:
    runner = TicketTriageRunner()
    task = TaskInput(
        task_id="ticket-001",
        scenario="ticket_triage",
        payload={"ticket_text": "用户投诉 5G 基站覆盖差，室内频繁掉线。"},
    )

    result = runner.run(task)

    assert result.output["label"] == "radio_access"
    assert result.output["troubleshooting_steps"]
    assert result.metrics["confidence"] > 0
    assert [step.name for step in result.trace.steps] == ["classify_ticket", "recommend_steps"]
