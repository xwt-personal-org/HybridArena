"""FastAPI app exposing AgentBench scenario runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from hybrid_arena.core.schema import TaskInput
from hybrid_arena.core.storage import AgentBenchStore
from hybrid_arena.scenarios.registry import get_runner, list_scenarios
from hybrid_arena.skill_runtime.adviser import SkillRuntimeAdviser
from hybrid_arena.skill_runtime.dispatcher import dispatch_workspace_event
from hybrid_arena.skill_runtime.memory import SkillMemoryStore
from hybrid_arena.skill_runtime.sample_skills import build_sample_skills
from hybrid_arena.skill_runtime.schema import PROTOCOL_VERSION, WorkspaceEvent
from hybrid_arena.skill_runtime.security import RuntimePermissionPolicy, SkillRuntimePermissionError
from hybrid_arena.skill_runtime.tool_registry import list_tools, summarize_policy

DEFAULT_DB_PATH = Path("results/agentbench/agentbench.db")
DEFAULT_SKILL_RUNTIME_ROOT = Path("results/agentbench/skill_runtime_workspace")


def create_app(
    store: AgentBenchStore | None = None,
    skill_runtime_db_path: Path | None = None,
    skill_runtime_root: Path | None = None,
    skill_runtime_allow_write: bool = False,
) -> FastAPI:
    app = FastAPI(title="HybridArena AgentBench API", version="0.3.0")
    app.state.store = store or AgentBenchStore(DEFAULT_DB_PATH)
    app.state.store.init_schema()
    runtime_root = Path(skill_runtime_root or DEFAULT_SKILL_RUNTIME_ROOT).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_db_path = Path(skill_runtime_db_path or (runtime_root / "state.db"))
    app.state.skill_runtime_root = runtime_root
    app.state.skill_runtime_skills = build_sample_skills()
    app.state.skill_runtime_policy = RuntimePermissionPolicy(
        allow_write_effects=skill_runtime_allow_write,
        workspace_root=runtime_root,
    )
    app.state.skill_runtime_memory = SkillMemoryStore(runtime_db_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "hybrid-arena-agentbench"}

    @app.get("/scenarios")
    def scenarios() -> dict[str, list[str]]:
        return {"scenarios": list_scenarios()}

    @app.post("/tasks/run")
    def run_task(payload: dict[str, Any]) -> dict[str, Any]:
        scenario = payload.get("scenario")
        try:
            runner = get_runner(scenario)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        task = TaskInput(
            task_id=payload["task_id"],
            scenario=scenario,
            payload=dict(payload.get("payload", {})),
            metadata=dict(payload.get("metadata", {})),
        )
        result = runner.run(task)
        app.state.store.save_run(result)
        return result.to_dict()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        result = app.state.store.get_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return result.to_dict()

    @app.get("/runs")
    def list_runs(scenario: str | None = None, limit: int = 50) -> dict[str, list[dict[str, Any]]]:
        runs = app.state.store.list_runs(scenario=scenario, limit=limit)
        return {"runs": [run.to_dict() for run in runs]}

    @app.get("/skill-runtime/tools")
    def skill_runtime_tools() -> dict[str, list[dict[str, Any]]]:
        return {
            "tools": list_tools(
                policy=app.state.skill_runtime_policy,
                skills=app.state.skill_runtime_skills,
            )
        }

    @app.get("/skill-runtime/policy")
    def skill_runtime_policy() -> dict[str, Any]:
        return summarize_policy(
            policy=app.state.skill_runtime_policy,
            skills=app.state.skill_runtime_skills,
        )

    @app.get("/skill-runtime/advice")
    def skill_runtime_advice() -> dict[str, list[dict[str, Any]]]:
        adviser = SkillRuntimeAdviser(app.state.skill_runtime_memory)
        return {
            "advice": adviser.advise(
                skills=app.state.skill_runtime_skills,
                policy=app.state.skill_runtime_policy,
            )
        }

    @app.post("/skill-runtime/dispatch")
    def skill_runtime_dispatch(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            event = parse_workspace_event(payload)
            result = dispatch_workspace_event(
                event=event,
                skills=app.state.skill_runtime_skills,
                policy=app.state.skill_runtime_policy,
                workspace_root=app.state.skill_runtime_root,
            )
        except SkillRuntimePermissionError as exc:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "skill_runtime_effect_denied",
                    "tool_id": exc.skill.id,
                    "reason": exc.reason,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if result is None:
            raise HTTPException(status_code=404, detail="No matching skill-runtime tool")
        return result.to_dict()

    return app


def parse_workspace_event(payload: dict[str, Any]) -> WorkspaceEvent:
    if not isinstance(payload, dict):
        raise ValueError("Workspace event payload must be a dictionary")
    version = payload.get("version")
    is_protocol_envelope = version == PROTOCOL_VERSION or (
        isinstance(version, str) and version.startswith("skill-runtime/")
    )
    if is_protocol_envelope:
        body = payload.get("body")
        if not isinstance(body, dict):
            raise ValueError("Malformed skill-runtime envelope")
        try:
            return WorkspaceEvent.from_dict(body)
        except ValueError as exc:
            raise ValueError("Malformed skill-runtime envelope") from exc
    return WorkspaceEvent.from_dict(payload)


app = create_app()
