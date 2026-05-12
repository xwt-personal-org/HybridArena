"""Tests for the reflex dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_arena.skill_runtime.body_schema import BodySchema
from hybrid_arena.skill_runtime.dispatcher import DispatchResult, ReflexDispatcher
from hybrid_arena.skill_runtime.sample_skills import create_sample_skills
from hybrid_arena.skill_runtime.schema import Effect, Skill, Trigger, TypedSignature, WorkspaceEvent
from hybrid_arena.skill_runtime.workspace import Workspace


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    return Workspace(root=tmp_path, db_path=tmp_path / "test.db")


@pytest.fixture()
def skills(ws: Workspace) -> list[Skill]:
    return create_sample_skills(ws)


@pytest.fixture()
def body(ws: Workspace, skills: list[Skill]) -> BodySchema:
    return BodySchema(skills=skills, workspace=ws)


@pytest.fixture()
def dispatcher(body: BodySchema, ws: Workspace) -> ReflexDispatcher:
    return ReflexDispatcher(body=body, workspace=ws)


class TestTriggerMatching:
    """Verify that events match expected skills."""

    def test_glob_matches_py_file(
        self, dispatcher: ReflexDispatcher
    ) -> None:
        event = WorkspaceEvent(kind="file_save", path="src/main.py")
        result = dispatcher.dispatch(event)
        assert result.skill_id == "format_on_save"
        assert result.success is True

    def test_glob_no_match_for_txt(
        self, dispatcher: ReflexDispatcher
    ) -> None:
        event = WorkspaceEvent(kind="file_save", path="README.txt")
        result = dispatcher.dispatch(event)
        # No glob match for *.txt → escalated
        assert result.escalated is True

    def test_regex_matches_rename_event(
        self, dispatcher: ReflexDispatcher, ws: Workspace
    ) -> None:
        fixture_path = ws.root / "tests" / "test_imports.py"
        fixture_path.parent.mkdir(parents=True)
        fixture_path.write_text(
            "from src.old_module import build\n",
            encoding="utf-8",
        )
        event = WorkspaceEvent(
            kind="file_rename",
            path="rename_utils",
            payload={
                "old_name": "src.old_module",
                "new_name": "src.new_module",
                "paths": ["tests/test_imports.py"],
            },
        )
        result = dispatcher.dispatch(event)
        assert result.skill_id == "update_imports_after_rename"
        assert result.success is True


class TestEscalation:
    """Verify escalation when no skill matches."""

    def test_unmatched_event_escalates(
        self, dispatcher: ReflexDispatcher
    ) -> None:
        event = WorkspaceEvent(kind="unknown_event", path="no_match.xyz")
        result = dispatcher.dispatch(event)
        assert result.escalated is True
        assert result.skill_id is None
        assert result.residual == 1.0

    def test_fallback_planner_called(
        self, body: BodySchema, ws: Workspace
    ) -> None:
        fallback_result = DispatchResult(
            skill_id="fallback",
            action={"fallback": True},
            escalated=False,
            success=True,
            residual=0.0,
            message="Fallback handled it.",
        )

        def fallback_planner(event: WorkspaceEvent) -> DispatchResult:
            return fallback_result

        dispatcher = ReflexDispatcher(
            body=body, workspace=ws, fallback_planner=fallback_planner
        )
        event = WorkspaceEvent(kind="unknown", path="no_match.xyz")
        result = dispatcher.dispatch(event)
        assert result.skill_id == "fallback"
        assert result.success is True


class TestTraceRecording:
    """Verify trace count increments on dispatch."""

    def test_trace_recorded_on_match(
        self, dispatcher: ReflexDispatcher, ws: Workspace
    ) -> None:
        before = len(ws.get_traces())
        event = WorkspaceEvent(kind="file_save", path="src/foo.py")
        dispatcher.dispatch(event)
        after = len(ws.get_traces())
        assert after == before + 1

    def test_trace_recorded_on_escalation(
        self, dispatcher: ReflexDispatcher, ws: Workspace
    ) -> None:
        before = len(ws.get_traces())
        event = WorkspaceEvent(kind="mystery", path="no_match")
        dispatcher.dispatch(event)
        after = len(ws.get_traces())
        assert after == before + 1

    def test_trace_skill_id(
        self, dispatcher: ReflexDispatcher, ws: Workspace
    ) -> None:
        event = WorkspaceEvent(kind="file_save", path="app.py")
        dispatcher.dispatch(event)
        traces = ws.get_traces()
        assert traces[0]["skill_id"] == "format_on_save"

    def test_unknown_controller_records_failure(self, ws: Workspace) -> None:
        skill = Skill(
            id="unknown_controller_skill",
            name="Unknown Controller Skill",
            triggers=(Trigger(kind="glob", spec="*.py"),),
            signature=TypedSignature(
                input_type="file_path",
                output_type="annotation_update",
                effects=frozenset({Effect.WRITE_FS}),
            ),
            controller="missing_controller",
        )
        body = BodySchema(skills=[skill], workspace=ws)
        dispatcher = ReflexDispatcher(body=body, workspace=ws)

        result = dispatcher.dispatch(WorkspaceEvent(kind="file_save", path="app.py"))

        assert result.skill_id == "unknown_controller_skill"
        assert result.success is False
        assert result.residual == 1.0
        assert "Unknown controller" in result.message
        traces = ws.get_traces(skill_id="unknown_controller_skill")
        assert traces[0]["success"] is False
        assert traces[0]["output_snapshot"]["error"] == "Unknown controller"


class TestBodySchemaSummary:
    """Verify BodySchema prompt summary lists affordances."""

    def test_summary_not_empty(
        self, body: BodySchema
    ) -> None:
        summary = body.to_prompt_summary()
        assert "Current affordances" in summary

    def test_summary_lists_skills(
        self, body: BodySchema
    ) -> None:
        summary = body.to_prompt_summary()
        assert "Format on Save" in summary

    def test_snapshot_structure(
        self, body: BodySchema
    ) -> None:
        snap = body.snapshot()
        assert "skill_count" in snap
        assert "affordance_count" in snap
        assert isinstance(snap["affordances"], list)
