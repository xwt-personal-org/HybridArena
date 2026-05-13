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
                "jd_text": "ะ่าช PythonกขFastAPI บอ RAGกฃ",
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


def test_skill_runtime_default_dispatch_denies_write_effect(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "state.db",
        skill_runtime_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.post(
        "/skill-runtime/dispatch",
        json={"kind": "event", "path": "notes/summary.md", "payload": {"action": "write_summary"}},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "skill_runtime_effect_denied"
    assert not (tmp_path / "workspace" / "notes" / "summary.md").exists()


def test_skill_runtime_dispatch_allows_explicit_write_opt_in(tmp_path) -> None:
    root = tmp_path / "workspace"
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "state.db",
        skill_runtime_root=root,
        skill_runtime_allow_write=True,
    )
    client = TestClient(app)

    response = client.post(
        "/skill-runtime/dispatch",
        json={"kind": "event", "path": "notes/summary.md", "payload": {"action": "write_summary"}},
    )

    assert response.status_code == 200
    assert response.json()["tool_id"] == "write_summary"
    assert (root / "notes" / "summary.md").read_text(encoding="utf-8").startswith(
        "# Skill Runtime Summary"
    )


def test_skill_runtime_tools_expose_policy(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "state.db",
        skill_runtime_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.get("/skill-runtime/tools")

    assert response.status_code == 200
    tools = {tool["id"]: tool for tool in response.json()["tools"]}
    assert tools["inspect_workspace"]["allowed"] is True
    assert tools["write_summary"]["allowed"] is False
    assert tools["write_summary"]["effects"] == ["READ_FS", "WRITE_FS"]
    assert "WRITE_FS" in tools["write_summary"]["blocked_reason"]


def test_skill_runtime_policy_endpoint_lists_blocked_tools(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "state.db",
        skill_runtime_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.get("/skill-runtime/policy")

    assert response.status_code == 200
    body = response.json()
    assert body["allow_write_effects"] is False
    assert body["allowed_tools"] == ["inspect_workspace"]
    assert {tool["id"] for tool in body["blocked_tools"]} == {
        "ask_llm",
        "fetch_url",
        "run_shell",
        "write_summary",
    }


def test_skill_runtime_advice_includes_policy_diagnostics(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "state.db",
        skill_runtime_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.get("/skill-runtime/advice")

    assert response.status_code == 200
    advice_ids = {item["id"] for item in response.json()["advice"]}
    assert "write_effects_disabled" in advice_ids
    assert "tool_policy_summary" in advice_ids


def test_plain_kind_event_payload_is_not_protocol_envelope() -> None:
    from hybrid_arena.services.api.app import parse_workspace_event

    event = parse_workspace_event({"kind": "event", "path": "src/app.py", "payload": {}})

    assert event.kind == "event"
    assert event.path == "src/app.py"
    assert event.payload == {}


def test_malformed_protocol_envelope_returns_http_400(tmp_path) -> None:
    app = create_app(
        AgentBenchStore(tmp_path / "agentbench.db"),
        skill_runtime_db_path=tmp_path / "state.db",
        skill_runtime_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.post(
        "/skill-runtime/dispatch",
        json={"version": "skill-runtime/1", "kind": "event", "body": {}},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Malformed skill-runtime envelope"
