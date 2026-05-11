"""Tests for SQLite-backed Workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_arena.skill_runtime.workspace import Workspace


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    """Provide a fresh workspace with a temporary database."""
    return Workspace(root=tmp_path, db_path=tmp_path / "test.db")


class TestAnnotations:
    """Annotation persistence and query tests."""

    def test_annotate_and_get(self, ws: Workspace) -> None:
        ws.annotate("src/foo.py", tags={"py", "needs_test"}, status="passing")
        ann = ws.get_annotation("src/foo.py")
        assert ann is not None
        assert ann.path == "src/foo.py"
        assert "py" in ann.tags
        assert "needs_test" in ann.tags
        assert ann.status == "passing"

    def test_get_missing_returns_none(self, ws: Workspace) -> None:
        assert ws.get_annotation("nonexistent.py") is None

    def test_annotate_overwrite(self, ws: Workspace) -> None:
        ws.annotate("src/foo.py", tags={"py"}, status="unknown")
        ws.annotate("src/foo.py", tags={"py", "formatted"}, status="passing")
        ann = ws.get_annotation("src/foo.py")
        assert ann is not None
        assert "formatted" in ann.tags
        assert ann.status == "passing"

    def test_persistence_after_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.db"
        ws1 = Workspace(root=tmp_path, db_path=db)
        ws1.annotate("src/bar.py", tags={"py"}, status="failing")
        # Re-open from same file.
        ws2 = Workspace(root=tmp_path, db_path=db)
        ann = ws2.get_annotation("src/bar.py")
        assert ann is not None
        assert ann.status == "failing"

    def test_lineage_and_decay(self, ws: Workspace) -> None:
        ws.annotate(
            "src/baz.py",
            tags={"py"},
            last_skill="format_on_save",
            decay_at=9999999999.0,
            lineage=("format_on_save", "add_test"),
        )
        ann = ws.get_annotation("src/baz.py")
        assert ann is not None
        assert ann.last_skill == "format_on_save"
        assert ann.decay_at == 9999999999.0
        assert ann.lineage == ("format_on_save", "add_test")


class TestQueryPaths:
    """query_paths predicate tests."""

    def test_tags_superset(self, ws: Workspace) -> None:
        ws.annotate("a.py", tags={"py", "needs_test"})
        ws.annotate("b.py", tags={"py"})
        ws.annotate("c.txt", tags={"doc"})
        result = ws.query_paths(tags_superset={"py", "needs_test"})
        assert result == ["a.py"]

    def test_status(self, ws: Workspace) -> None:
        ws.annotate("x.py", status="failing")
        ws.annotate("y.py", status="passing")
        result = ws.query_paths(status="failing")
        assert result == ["x.py"]

    def test_any_tag(self, ws: Workspace) -> None:
        ws.annotate("p.py", tags={"py", "formatted"})
        ws.annotate("q.py", tags={"py"})
        ws.annotate("r.txt", tags={"doc"})
        result = ws.query_paths(any_tag="formatted")
        assert result == ["p.py"]

    def test_no_predicate_returns_empty(self, ws: Workspace) -> None:
        ws.annotate("a.py", tags={"py"})
        assert ws.query_paths() == []


class TestEvents:
    """Event recording tests."""

    def test_record_event(self, ws: Workspace) -> None:
        ws.record_event("file_save", path="src/foo.py", payload={"size": 123})
        # Events are write-only in L0; just verify no exception.
        # We can inspect via direct SQL for verification.
        import sqlite3

        conn = sqlite3.connect(str(ws.db_path))
        row = conn.execute(
            "SELECT kind, path, payload FROM events"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "file_save"
        assert row[1] == "src/foo.py"


class TestTraces:
    """Trace recording and retrieval tests."""

    def test_record_and_get_trace(self, ws: Workspace) -> None:
        ws.record_trace(
            skill_id="format_on_save",
            event_kind="file_save",
            input_snapshot={"path": "foo.py"},
            output_snapshot={"formatted": True},
            success=True,
            residual=0.0,
        )
        traces = ws.get_traces(skill_id="format_on_save")
        assert len(traces) == 1
        assert traces[0]["skill_id"] == "format_on_save"
        assert traces[0]["success"] is True

    def test_get_traces_limit(self, ws: Workspace) -> None:
        for i in range(5):
            ws.record_trace(
                skill_id=f"skill_{i}",
                event_kind="test",
                input_snapshot={},
                output_snapshot={},
                success=True,
                residual=0.0,
            )
        traces = ws.get_traces(limit=3)
        assert len(traces) == 3

    def test_get_traces_all(self, ws: Workspace) -> None:
        ws.record_trace("s1", "e1", {}, {}, True, 0.0)
        ws.record_trace("s2", "e2", {}, {}, False, 0.5)
        all_traces = ws.get_traces()
        assert len(all_traces) == 2


class TestBodySnapshot:
    """Body-schema snapshot persistence."""

    def test_save_body_snapshot(self, ws: Workspace) -> None:
        ws.save_body_snapshot('{"skill_count": 5}')
        # Verify via direct SQL.
        import sqlite3

        conn = sqlite3.connect(str(ws.db_path))
        row = conn.execute(
            "SELECT snapshot FROM body_schema_snapshots"
        ).fetchone()
        conn.close()
        assert row is not None
        assert "skill_count" in row[0]
