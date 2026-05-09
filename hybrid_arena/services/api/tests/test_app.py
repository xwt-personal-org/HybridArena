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
