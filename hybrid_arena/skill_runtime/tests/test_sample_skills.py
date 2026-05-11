"""Tests for sample skills creation and dispatch integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_arena.skill_runtime.body_schema import BodySchema
from hybrid_arena.skill_runtime.dispatcher import ReflexDispatcher
from hybrid_arena.skill_runtime.sample_skills import create_sample_skills
from hybrid_arena.skill_runtime.schema import WorkspaceEvent
from hybrid_arena.skill_runtime.workspace import Workspace


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    return Workspace(root=tmp_path, db_path=tmp_path / "test.db")


class TestSampleSkillCreation:
    """Verify sample skills are created correctly."""

    def test_returns_five_skills(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        assert len(skills) == 5

    def test_expected_skill_ids(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        ids = {s.id for s in skills}
        assert ids == {
            "format_on_save",
            "add_test_for_new_function",
            "fix_failing_test",
            "update_imports_after_rename",
            "escalate_when_stuck",
        }

    def test_format_on_save_triggers(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        fmt = next(s for s in skills if s.id == "format_on_save")
        assert len(fmt.triggers) == 1
        assert fmt.triggers[0].kind == "glob"
        assert fmt.triggers[0].spec == "*.py"

    def test_fix_failing_test_triggers(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        fix = next(s for s in skills if s.id == "fix_failing_test")
        assert len(fix.triggers) == 1
        assert fix.triggers[0].kind == "annotation"
        assert fix.triggers[0].spec == "failing"

    def test_update_imports_triggers(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        upd = next(s for s in skills if s.id == "update_imports_after_rename")
        assert len(upd.triggers) == 1
        assert upd.triggers[0].kind == "regex"

    def test_escalate_when_stuck_has_no_triggers(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        esc = next(s for s in skills if s.id == "escalate_when_stuck")
        assert len(esc.triggers) == 0


class TestSampleSkillDispatch:
    """End-to-end dispatch with sample skills."""

    def test_format_on_save_dispatch(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        body = BodySchema(skills=skills, workspace=ws)
        dispatcher = ReflexDispatcher(body=body, workspace=ws)
        event = WorkspaceEvent(kind="file_save", path="src/app.py")
        result = dispatcher.dispatch(event)
        assert result.skill_id == "format_on_save"
        assert result.success is True

    def test_no_match_escalates(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        body = BodySchema(skills=skills, workspace=ws)
        dispatcher = ReflexDispatcher(body=body, workspace=ws)
        event = WorkspaceEvent(kind="file_save", path="README.md")
        result = dispatcher.dispatch(event)
        assert result.escalated is True

    def test_trace_count_after_dispatch(self, ws: Workspace) -> None:
        skills = create_sample_skills(ws)
        body = BodySchema(skills=skills, workspace=ws)
        dispatcher = ReflexDispatcher(body=body, workspace=ws)
        event = WorkspaceEvent(kind="file_save", path="main.py")
        dispatcher.dispatch(event)
        assert len(ws.get_traces()) == 1
