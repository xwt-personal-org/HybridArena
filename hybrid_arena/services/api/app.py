"""FastAPI app exposing AgentBench scenario runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from hybrid_arena.core.schema import TaskInput
from hybrid_arena.core.storage import AgentBenchStore
from hybrid_arena.scenarios.registry import get_runner, list_scenarios
from hybrid_arena.skill_runtime.adviser import SkillRuntimeAdviser
from hybrid_arena.skill_runtime.body_schema import BodySchema
from hybrid_arena.skill_runtime.dispatcher import ReflexDispatcher
from hybrid_arena.skill_runtime.memory import SkillMemoryStore
from hybrid_arena.skill_runtime.protocol import EnvelopeKind, SkillRuntimeMessage
from hybrid_arena.skill_runtime.sample_skills import create_sample_skills
from hybrid_arena.skill_runtime.schema import WorkspaceEvent
from hybrid_arena.skill_runtime.tool_registry import ToolRegistry
from hybrid_arena.skill_runtime.workspace import Workspace

DEFAULT_DB_PATH = Path("results/agentbench/agentbench.db")
DEFAULT_SKILL_RUNTIME_DB_PATH = Path("results/agentbench/skill_runtime.db")


def create_app(
    store: AgentBenchStore | None = None,
    skill_runtime_db_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="HybridArena AgentBench API", version="0.3.0")
    app.state.store = store or AgentBenchStore(DEFAULT_DB_PATH)
    app.state.store.init_schema()
    app.state.skill_runtime_db_path = skill_runtime_db_path or DEFAULT_SKILL_RUNTIME_DB_PATH

    def skill_runtime_context() -> tuple[Workspace, BodySchema, ReflexDispatcher]:
        workspace = Workspace(
            root=Path(".").resolve(),
            db_path=app.state.skill_runtime_db_path,
        )
        memory = SkillMemoryStore(app.state.skill_runtime_db_path)
        body = BodySchema(
            skills=create_sample_skills(workspace),
            workspace=workspace,
        )
        dispatcher = ReflexDispatcher(
            body=body,
            workspace=workspace,
            memory=memory,
        )
        return workspace, body, dispatcher

    def parse_workspace_event(payload: dict[str, Any]) -> WorkspaceEvent:
        if payload.get("version") or (payload.get("kind") == EnvelopeKind.EVENT.value and "payload" in payload):
            return SkillRuntimeMessage.from_dict(payload).to_workspace_event()
        return WorkspaceEvent(
            kind=str(payload["kind"]),
            path=str(payload.get("path", "")),
            payload=dict(payload.get("payload", {})),
            created_at=float(payload.get("created_at", 0.0)),
        )

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
        registry = ToolRegistry.discover_builtin_controllers()
        return {"tools": [tool.to_dict() for tool in registry.list()]}

    @app.get("/skill-runtime/advice")
    def skill_runtime_advice() -> dict[str, list[dict[str, Any]]]:
        workspace, _body, _dispatcher = skill_runtime_context()
        memory = SkillMemoryStore(app.state.skill_runtime_db_path)
        advisories = SkillRuntimeAdviser(workspace, memory=memory).advise()
        return {"advisories": [advisory.to_dict() for advisory in advisories]}

    @app.post("/skill-runtime/dispatch")
    def skill_runtime_dispatch(payload: dict[str, Any]) -> dict[str, Any]:
        workspace, body, dispatcher = skill_runtime_context()
        try:
            event = parse_workspace_event(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = dispatcher.dispatch(event)
        return {
            "result": {
                "skill_id": result.skill_id,
                "action": result.action,
                "escalated": result.escalated,
                "success": result.success,
                "residual": result.residual,
                "message": result.message,
            },
            "affordances": body.explain_affordances(event=event),
            "trace_count": len(workspace.get_traces()),
        }

    return app


app = create_app()
