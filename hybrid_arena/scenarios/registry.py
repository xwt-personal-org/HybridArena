"""Scenario registry for AgentBench runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

from hybrid_arena.core.schema import TaskInput, TaskRunResult


class ScenarioRunner(Protocol):
    scenario_name: str

    def run(self, task: TaskInput) -> TaskRunResult:
        ...


_RUNNER_IMPORTS = {
    "jd_resume_match": "hybrid_arena.scenarios.jd_resume_match.runner:JDResumeMatchRunner",
    "telecom_rag": "hybrid_arena.scenarios.telecom_rag.runner:TelecomRagRunner",
    "ticket_triage": "hybrid_arena.scenarios.ticket_triage.runner:TicketTriageRunner",
}


def list_scenarios() -> list[str]:
    return list(_RUNNER_IMPORTS)


def get_runner(scenario_name: str, **kwargs: Any) -> ScenarioRunner:
    import_path = _RUNNER_IMPORTS.get(scenario_name)
    if import_path is None:
        raise KeyError(f"Unknown scenario: {scenario_name}")
    module_name, class_name = import_path.split(":")
    module = import_module(module_name)
    runner_cls = getattr(module, class_name)
    return runner_cls(**kwargs)
