"""Runner for network ticket triage."""

from __future__ import annotations

from hybrid_arena.core.schema import TaskInput, TaskRunResult
from hybrid_arena.core.traces import TraceRecorder
from hybrid_arena.scenarios.ticket_triage.classifier import classify_ticket
from hybrid_arena.scenarios.ticket_triage.recommender import recommend_troubleshooting_steps


class TicketTriageRunner:
    scenario_name = "ticket_triage"

    def run(self, task: TaskInput) -> TaskRunResult:
        run_id = task.metadata.get("run_id", f"{task.task_id}-ticket-run")
        ticket_text = task.payload["ticket_text"]
        recorder = TraceRecorder(run_id=run_id, task_id=task.task_id, scenario=self.scenario_name)

        recorder.start_step("classify_ticket", {"ticket_text": ticket_text})
        prediction = classify_ticket(ticket_text)
        recorder.finish_step(prediction.to_dict())

        recorder.start_step("recommend_steps", {"label": prediction.label})
        troubleshooting_steps = recommend_troubleshooting_steps(prediction.label)
        output = {
            **prediction.to_dict(),
            "troubleshooting_steps": troubleshooting_steps,
        }
        metrics = {"confidence": prediction.confidence, "unknown": prediction.label == "unknown"}
        recorder.finish_step({"troubleshooting_steps": troubleshooting_steps}, metrics=metrics)

        return TaskRunResult(
            run_id=run_id,
            task_id=task.task_id,
            scenario=self.scenario_name,
            output=output,
            metrics=metrics,
            trace=recorder.to_trace(metrics=metrics),
        )
