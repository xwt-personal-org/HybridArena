from __future__ import annotations

from fastapi.testclient import TestClient

from hybrid_arena.core.storage import AgentBenchStore
from hybrid_arena.services.api.app import create_app


def test_health_and_scenarios(tmp_path) -> None:
    app = create_app(AgentBenchStore(tmp_path / "agentbench.db"))
    client = TestClient(app)

    health = client.get("/health")
    scenarios = client.get("/scenarios")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "hybrid-arena-agentbench"}
    assert scenarios.status_code == 200
    assert "jd_resume_match" in scenarios.json()["scenarios"]


def test_run_task_persists_result_and_can_fetch_it(tmp_path) -> None:
    app = create_app(AgentBenchStore(tmp_path / "agentbench.db"))
    client = TestClient(app)

    run_response = client.post(
        "/tasks/run",
        json={
            "task_id": "jd-001",
            "scenario": "jd_resume_match",
            "payload": {
                "jd_text": "需要 Python、FastAPI 和 RAG。",
                "resume_profile": {"skills": ["python_backend"], "evidence": {}},
            },
            "metadata": {"run_id": "run-001"},
        },
    )
    fetch_response = client.get("/runs/run-001")
    list_response = client.get("/runs")

    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == "run-001"
    assert fetch_response.status_code == 200
    assert fetch_response.json()["trace"]["steps"][0]["name"] == "extract_jd_requirements"
    assert list_response.json()["runs"][0]["run_id"] == "run-001"


def test_run_task_rejects_unknown_scenario(tmp_path) -> None:
    app = create_app(AgentBenchStore(tmp_path / "agentbench.db"))
    client = TestClient(app)

    response = client.post(
        "/tasks/run",
        json={"task_id": "bad-001", "scenario": "unknown", "payload": {}, "metadata": {}},
    )

    assert response.status_code == 404


def test_skill_runtime_tools_endpoint(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "skill_runtime.db",
    )
    client = TestClient(app)

    response = client.get("/skill-runtime/tools")

    assert response.status_code == 200
    assert any(
        item["name"] == "mock_annotate_formatted"
        for item in response.json()["tools"]
    )


def test_skill_runtime_advice_endpoint_reports_empty_workspace(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "skill_runtime.db",
    )
    client = TestClient(app)

    response = client.get("/skill-runtime/advice")

    assert response.status_code == 200
    assert any(
        item["kind"] == "empty_workspace"
        for item in response.json()["advisories"]
    )


def test_skill_runtime_dispatch_endpoint_runs_deterministic_skill(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "skill_runtime.db",
    )
    client = TestClient(app)

    response = client.post(
        "/skill-runtime/dispatch",
        json={"kind": "file_save", "path": "src/app.py", "payload": {}},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["result"]["skill_id"] == "format_on_save"
    assert body["result"]["success"] is True
    assert body["trace_count"] == 1
