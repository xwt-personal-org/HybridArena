"""Tests for deterministic skill-runtime advisories."""

from __future__ import annotations

from pathlib import Path

from hybrid_arena.skill_runtime.adviser import SkillRuntimeAdviser
from hybrid_arena.skill_runtime.workspace import Workspace


def test_adviser_reports_empty_workspace(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")

    advisories = SkillRuntimeAdviser(workspace).advise(now=100.0)

    assert any(item.kind == "empty_workspace" for item in advisories)


def test_adviser_reports_repeated_escalation(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    for _ in range(3):
        workspace.record_trace("__escalated__", "unknown", {}, {"escalated": True}, False, 1.0)

    advisories = SkillRuntimeAdviser(workspace).advise(now=100.0)

    assert any(item.kind == "repeated_escalation" for item in advisories)


def test_adviser_reports_low_success_skill_and_stale_annotation(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, db_path=tmp_path / "runtime.db")
    workspace.annotate("stale.py", tags={"py"}, decay_at=10.0)
    workspace.record_trace("fragile_skill", "file_save", {}, {}, False, 1.0)
    workspace.record_trace("fragile_skill", "file_save", {}, {}, True, 0.0)

    advisories = SkillRuntimeAdviser(workspace).advise(now=100.0)
    kinds = {item.kind for item in advisories}

    assert "low_success_skill" in kinds
    assert "stale_annotations" in kinds
