"""FastAPI app exposing AgentBench scenario runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from hybrid_arena.core.schema import TaskInput
from hybrid_arena.core.storage import AgentBenchStore
from hybrid_arena.scenarios.registry import get_runner, list_scenarios

DEFAULT_DB_PATH = Path("results/agentbench/agentbench.db")


def create_app(store: AgentBenchStore | None = None) -> FastAPI:
    app = FastAPI(title="HybridArena AgentBench API", version="0.3.0")
    app.state.store = store or AgentBenchStore(DEFAULT_DB_PATH)
    app.state.store.init_schema()

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

    return app


app = create_app()
