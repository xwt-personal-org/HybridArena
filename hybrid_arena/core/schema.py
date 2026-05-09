"""Shared AgentBench data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    scenario: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskInput:
        return cls(
            task_id=data["task_id"],
            scenario=data["scenario"],
            payload=dict(data.get("payload", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallRecord:
        return cls(
            name=data["name"],
            input=dict(data.get("input", {})),
            output=dict(data.get("output", {})),
            latency_ms=float(data.get("latency_ms", 0.0)),
            success=bool(data.get("success", False)),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class TaskTrace:
    run_id: str
    task_id: str
    scenario: str
    steps: list[ToolCallRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "scenario": self.scenario,
            "steps": [step.to_dict() for step in self.steps],
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTrace:
        return cls(
            run_id=data["run_id"],
            task_id=data["task_id"],
            scenario=data["scenario"],
            steps=[ToolCallRecord.from_dict(step) for step in data.get("steps", [])],
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(frozen=True)
class TaskRunResult:
    run_id: str
    task_id: str
    scenario: str
    output: dict[str, Any]
    metrics: dict[str, Any]
    trace: TaskTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "scenario": self.scenario,
            "output": self.output,
            "metrics": self.metrics,
            "trace": self.trace.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRunResult:
        return cls(
            run_id=data["run_id"],
            task_id=data["task_id"],
            scenario=data["scenario"],
            output=dict(data.get("output", {})),
            metrics=dict(data.get("metrics", {})),
            trace=TaskTrace.from_dict(data["trace"]),
        )


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    total: int
    metrics: dict[str, Any]
    cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        return cls(
            scenario=data["scenario"],
            total=int(data["total"]),
            metrics=dict(data.get("metrics", {})),
            cases=[dict(case) for case in data.get("cases", [])],
        )
