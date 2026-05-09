"""Trace recording for AgentBench task runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from hybrid_arena.core.schema import TaskTrace, ToolCallRecord


class TraceRecorder:
    """Records deterministic task steps without depending on an external tracing service."""

    def __init__(self, run_id: str, task_id: str, scenario: str):
        self.run_id = run_id
        self.task_id = task_id
        self.scenario = scenario
        self._steps: list[ToolCallRecord] = []
        self._current_name: str | None = None
        self._current_input: dict[str, Any] | None = None
        self._current_started_at: float | None = None

    def start_step(self, name: str, payload: dict[str, Any]) -> None:
        if self._current_name is not None:
            raise RuntimeError(f"Trace step {self._current_name!r} is already open")
        self._current_name = name
        self._current_input = dict(payload)
        self._current_started_at = time.perf_counter()

    def finish_step(
        self,
        output: dict[str, Any],
        *,
        success: bool = True,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self._current_name is None or self._current_started_at is None:
            raise RuntimeError("No trace step is open")
        latency_ms = (time.perf_counter() - self._current_started_at) * 1000
        step_output = dict(output)
        if metrics:
            step_output["metrics"] = dict(metrics)
        self._steps.append(
            ToolCallRecord(
                name=self._current_name,
                input=self._current_input or {},
                output=step_output,
                latency_ms=latency_ms,
                success=success,
                error=error,
            )
        )
        self._current_name = None
        self._current_input = None
        self._current_started_at = None

    def to_trace(self, metrics: dict[str, Any] | None = None) -> TaskTrace:
        if self._current_name is not None:
            raise RuntimeError(f"Trace step {self._current_name!r} has not been finished")
        return TaskTrace(
            run_id=self.run_id,
            task_id=self.task_id,
            scenario=self.scenario,
            steps=list(self._steps),
            metrics=dict(metrics or {}),
        )


class JsonlTraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: TaskTrace) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
