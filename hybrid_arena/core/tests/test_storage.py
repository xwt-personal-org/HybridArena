from __future__ import annotations

from hybrid_arena.core.schema import TaskRunResult, TaskTrace
from hybrid_arena.core.storage import AgentBenchStore


def test_store_saves_and_loads_run(tmp_path) -> None:
    store = AgentBenchStore(tmp_path / "agentbench.db")
    store.init_schema()
    result = TaskRunResult(
        run_id="run-001",
        task_id="task-001",
        scenario="jd_resume_match",
        output={"summary": "ok"},
        metrics={"score": 1.0},
        trace=TaskTrace(run_id="run-001", task_id="task-001", scenario="jd_resume_match"),
    )

    store.save_run(result)
    loaded = store.get_run("run-001")

    assert loaded == result


def test_store_lists_runs_by_scenario(tmp_path) -> None:
    store = AgentBenchStore(tmp_path / "agentbench.db")
    store.init_schema()
    store.save_run(
        TaskRunResult(
            run_id="run-001",
            task_id="task-001",
            scenario="jd_resume_match",
            output={},
            metrics={},
            trace=TaskTrace(run_id="run-001", task_id="task-001", scenario="jd_resume_match"),
        )
    )
    store.save_run(
        TaskRunResult(
            run_id="run-002",
            task_id="task-002",
            scenario="telecom_rag",
            output={},
            metrics={},
            trace=TaskTrace(run_id="run-002", task_id="task-002", scenario="telecom_rag"),
        )
    )

    runs = store.list_runs(scenario="telecom_rag")

    assert [run.run_id for run in runs] == ["run-002"]
