"""Tests for skill-runtime memory and workspace annotation helpers."""

from __future__ import annotations

from pathlib import Path

from hybrid_arena.skill_runtime.memory import SkillMemoryRecord, SkillMemoryStore
from hybrid_arena.skill_runtime.workspace import Workspace


def test_memory_store_upsert_get_and_query_by_skill_and_tags(tmp_path: Path) -> None:
    store = SkillMemoryStore(tmp_path / "memory.db")
    record = SkillMemoryRecord(
        skill_id="format_on_save",
        key="src/app.py",
        value={"status": "passing"},
        tags=frozenset({"py", "formatted"}),
        confidence=0.9,
        metadata={"source": "unit"},
        updated_at=100.0,
    )

    store.upsert(record)
    restored = store.get("format_on_save", "src/app.py")
    queried = store.query(skill_id="format_on_save", tags_superset={"py"})

    assert restored == record
    assert queried == [record]


def test_memory_store_decay_reduces_expired_confidence(tmp_path: Path) -> None:
    store = SkillMemoryStore(tmp_path / "memory.db")
    store.upsert(
        SkillMemoryRecord(
            skill_id="fix_failing_test",
            key="tests/test_app.py",
            value={"status": "failing"},
            confidence=1.0,
            decay_at=10.0,
        )
    )

    decayed = store.decay(now=20.0, factor=0.5)
    restored = store.get("fix_failing_test", "tests/test_app.py")

    assert decayed == 1
    assert restored is not None
    assert restored.confidence == 0.5
    assert restored.decay_at == 0.0


def test_memory_store_summarizes_workspace_traces(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    workspace.record_trace("format_on_save", "file_save", {}, {}, True, 0.0)
    workspace.record_trace("format_on_save", "file_save", {}, {}, False, 1.0)
    store = SkillMemoryStore(tmp_path / "runtime.db")

    summary = store.summarize_traces(skill_id="format_on_save")

    assert summary["total"] == 2
    assert summary["successes"] == 1
    assert summary["failures"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["avg_residual"] == 0.5


def test_workspace_query_paths_supports_glob_and_combined_tags(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    workspace.annotate("src/app.py", tags={"py", "needs_test"})
    workspace.annotate("src/app.md", tags={"doc", "needs_test"})
    workspace.annotate("tests/test_app.py", tags={"py", "generated"})

    result = workspace.query_paths(path_glob="src/*.py", tags_superset={"py"})

    assert result == ["src/app.py"]


def test_workspace_snapshots_and_prunes_expired_annotations(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    workspace.annotate("fresh.py", tags={"py"}, decay_at=100.0)
    workspace.annotate("expired.py", tags={"py"}, decay_at=10.0)

    before = workspace.snapshot_annotations()
    removed = workspace.prune_expired_annotations(now=50.0)
    after = workspace.snapshot_annotations()

    assert [item.path for item in before] == ["expired.py", "fresh.py"]
    assert removed == 1
    assert [item.path for item in after] == ["fresh.py"]
