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


def _dispatcher_for_skill(ws: Workspace, skill_id: str) -> ReflexDispatcher:
    skills = [s for s in create_sample_skills(ws) if s.id == skill_id]
    body = BodySchema(skills=skills, workspace=ws)
    return ReflexDispatcher(body=body, workspace=ws)


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

    def test_add_test_for_new_function_creates_pytest_file(
        self, ws: Workspace
    ) -> None:
        source_path = "src/calculator.py"
        ws.annotate(
            path=source_path,
            tags={"needs_test", "py"},
            status="unknown",
        )
        dispatcher = _dispatcher_for_skill(ws, "add_test_for_new_function")

        result = dispatcher.dispatch(
            WorkspaceEvent(kind="annotation_update", path=source_path)
        )

        test_path = ws.root / "tests" / "test_calculator.py"
        assert result.success is True
        assert test_path.exists()
        assert "def test_calculator_placeholder" in test_path.read_text()
        annotation = ws.get_annotation("tests/test_calculator.py")
        assert annotation is not None
        assert annotation.tags == frozenset({"generated", "test_file"})
        assert annotation.status == "passing"

    def test_fix_failing_test_toggles_annotation_to_passing(
        self, ws: Workspace
    ) -> None:
        failing_path = "tests/test_calculator.py"
        ws.annotate(path=failing_path, tags={"test_file"}, status="failing")
        dispatcher = _dispatcher_for_skill(ws, "fix_failing_test")

        result = dispatcher.dispatch(
            WorkspaceEvent(kind="test_fail", path=failing_path)
        )

        annotation = ws.get_annotation(failing_path)
        assert result.success is True
        assert annotation is not None
        assert annotation.status == "passing"
        assert annotation.last_skill == "fix_failing_test"

    def test_update_imports_after_rename_modifies_fixture_file(
        self, ws: Workspace
    ) -> None:
        fixture_path = ws.root / "tests" / "test_imports.py"
        fixture_path.parent.mkdir(parents=True)
        fixture_path.write_text(
            "from src.old_module import build\n\n"
            "def test_build():\n"
            "    assert build() is not None\n",
            encoding="utf-8",
        )
        dispatcher = _dispatcher_for_skill(ws, "update_imports_after_rename")

        result = dispatcher.dispatch(
            WorkspaceEvent(
                kind="file_rename",
                path="rename_old_module",
                payload={
                    "old_name": "src.old_module",
                    "new_name": "src.new_module",
                    "paths": ["tests/test_imports.py"],
                },
            )
        )

        assert result.success is True
        assert "src.new_module" in fixture_path.read_text(encoding="utf-8")
        assert "src.old_module" not in fixture_path.read_text(encoding="utf-8")

    def test_dispatcher_trace_records_controller_output(
        self, ws: Workspace
    ) -> None:
        dispatcher = _dispatcher_for_skill(ws, "format_on_save")

        dispatcher.dispatch(WorkspaceEvent(kind="file_save", path="src/app.py"))

        trace = ws.get_traces(skill_id="format_on_save")[0]
        assert trace["output_snapshot"]["controller"] == "mock_annotate_formatted"
        assert trace["output_snapshot"]["path"] == "src/app.py"
        assert trace["output_snapshot"]["tags"] == ["formatted", "py"]
        assert trace["output_snapshot"]["status"] == "passing"
