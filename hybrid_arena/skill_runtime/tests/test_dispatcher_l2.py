"""Tests for L2-lite dispatcher diagnostics and trace envelopes."""

from __future__ import annotations

from pathlib import Path

from hybrid_arena.skill_runtime.body_schema import BodySchema
from hybrid_arena.skill_runtime.dispatcher import DispatchPolicy, ReflexDispatcher
from hybrid_arena.skill_runtime.memory import SkillMemoryStore
from hybrid_arena.skill_runtime.sample_skills import create_sample_skills
from hybrid_arena.skill_runtime.schema import WorkspaceEvent
from hybrid_arena.skill_runtime.workspace import Workspace


def test_dispatch_policy_controls_bid_boundaries(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    skills = create_sample_skills(workspace)
    body = BodySchema(skills=skills, workspace=workspace)
    dispatcher = ReflexDispatcher(
        body=body,
        workspace=workspace,
        policy=DispatchPolicy(tonic_inhibition=0.2, no_go_weight=0.25),
    )
    skill = next(item for item in skills if item.id == "format_on_save")

    assert dispatcher.trigger_score(skill.triggers[0], WorkspaceEvent("file_save", "x.py")) == 1.0
    assert dispatcher.tonic_inhibition(skill) == 0.2
    assert dispatcher.no_go_penalty(skill) == 0.0


def test_body_schema_explains_affordances_for_event_path_annotation(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    workspace.annotate("src/calculator.py", tags={"py", "needs_test"})
    body = BodySchema(skills=create_sample_skills(workspace), workspace=workspace)

    diagnostics = body.explain_affordances(
        event=WorkspaceEvent(kind="annotation_update", path="src/calculator.py"),
        top_k=5,
    )
    add_test = next(item for item in diagnostics if item["id"] == "add_test_for_new_function")

    assert add_test["event"]["path"] == "src/calculator.py"
    assert add_test["applicable"] is True
    assert add_test["missing_preconditions"] == []
    assert add_test["triggers"][0]["event_path_match"] is True


def test_dispatch_records_protocol_envelope_metadata_in_trace(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    body = BodySchema(skills=create_sample_skills(workspace), workspace=workspace)
    dispatcher = ReflexDispatcher(
        body=body,
        workspace=workspace,
        memory=SkillMemoryStore(tmp_path / "runtime.db"),
    )

    dispatcher.dispatch(WorkspaceEvent(kind="file_save", path="src/app.py"))
    trace = workspace.get_traces(skill_id="format_on_save")[0]

    assert trace["input_snapshot"]["envelope"]["envelope_kind"] == "event"
    assert trace["input_snapshot"]["envelope"]["metadata"]["path"] == "src/app.py"
    assert trace["output_snapshot"]["envelope"]["envelope_kind"] == "result"
    assert trace["output_snapshot"]["controller"] == "mock_annotate_formatted"


def test_event_path_annotation_dispatches_before_global_annotation_fallback(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    workspace.annotate("src/local.py", tags={"py", "needs_test"})
    workspace.annotate("src/global.py", tags={"py", "needs_test"})
    skill = next(
        item for item in create_sample_skills(workspace) if item.id == "add_test_for_new_function"
    )
    dispatcher = ReflexDispatcher(
        body=BodySchema(skills=[skill], workspace=workspace),
        workspace=workspace,
    )

    result = dispatcher.dispatch(
        WorkspaceEvent(kind="annotation_update", path="src/local.py")
    )

    assert result.success is True
    assert result.action["source_path"] == "src/local.py"
    assert (tmp_path / "tests" / "test_local.py").exists()
    assert not (tmp_path / "tests" / "test_global.py").exists()
