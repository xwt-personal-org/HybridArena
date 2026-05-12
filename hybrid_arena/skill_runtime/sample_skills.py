"""Deterministic sample skills for the L0/L1 prototype.

No LLM or external API calls — every skill is a pure, deterministic mock
that exercises the dispatch→trace pipeline.
"""

from __future__ import annotations

from pathlib import Path

from hybrid_arena.skill_runtime.dispatcher import register_controller
from hybrid_arena.skill_runtime.schema import (
    Effect,
    Skill,
    Trigger,
    TypedSignature,
    WorkspaceEvent,
)
from hybrid_arena.skill_runtime.workspace import Workspace


def _safe_workspace_path(workspace: Workspace, relative_path: str) -> Path:
    root = workspace.root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes workspace root: {relative_path}")
    return candidate


def _mock_annotate_formatted(
    skill: Skill,
    event: WorkspaceEvent,
    workspace: Workspace,
) -> dict:
    workspace.annotate(
        path=event.path,
        tags={"formatted", "py"},
        status="passing",
        last_skill=skill.id,
        lineage=(skill.id,),
    )
    return {
        "success": True,
        "controller": skill.controller,
        "path": event.path,
        "tags": ["formatted", "py"],
        "status": "passing",
        "message": f"Annotated {event.path} as formatted.",
    }


def _mock_create_pytest_skeleton(
    skill: Skill,
    event: WorkspaceEvent,
    workspace: Workspace,
) -> dict:
    if not event.path:
        return {
            "success": False,
            "controller": skill.controller,
            "error": "Missing event path",
        }
    source_stem = Path(event.path).stem
    test_rel = Path("tests") / f"test_{source_stem}.py"
    test_path = _safe_workspace_path(workspace, test_rel.as_posix())
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_name = source_stem.replace("-", "_")
    test_path.write_text(
        f"def test_{test_name}_placeholder():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    annotation_path = test_rel.as_posix()
    workspace.annotate(
        path=annotation_path,
        tags={"generated", "test_file"},
        status="passing",
        last_skill=skill.id,
        lineage=(skill.id,),
    )
    return {
        "success": True,
        "controller": skill.controller,
        "source_path": event.path,
        "test_path": annotation_path,
        "tags": ["generated", "test_file"],
        "status": "passing",
        "message": f"Created {annotation_path}.",
    }


def _mock_toggle_to_passing(
    skill: Skill,
    event: WorkspaceEvent,
    workspace: Workspace,
) -> dict:
    candidate_paths = [event.path] if event.path else []
    candidate_paths.extend(
        p for p in workspace.query_paths(status="failing") if p not in candidate_paths
    )
    for path in candidate_paths:
        annotation = workspace.get_annotation(path)
        if annotation is None or annotation.status != "failing":
            continue
        workspace.annotate(
            path=annotation.path,
            tags=set(annotation.tags),
            status="passing",
            last_skill=skill.id,
            decay_at=annotation.decay_at,
            lineage=annotation.lineage + (skill.id,),
        )
        return {
            "success": True,
            "controller": skill.controller,
            "path": annotation.path,
            "status": "passing",
            "message": f"Marked {annotation.path} as passing.",
        }
    return {
        "success": False,
        "controller": skill.controller,
        "error": "No failing annotation found",
    }


def _mock_update_imports(
    skill: Skill,
    event: WorkspaceEvent,
    workspace: Workspace,
) -> dict:
    old_name = event.payload.get("old_name")
    new_name = event.payload.get("new_name")
    paths = event.payload.get("paths", ())
    if not old_name or not new_name or not paths:
        return {
            "success": False,
            "controller": skill.controller,
            "error": "old_name, new_name and paths are required",
        }

    updated_paths: list[str] = []
    for relative_path in paths:
        file_path = _safe_workspace_path(workspace, str(relative_path))
        if not file_path.exists() or not file_path.is_file():
            continue
        content = file_path.read_text(encoding="utf-8")
        updated = content.replace(str(old_name), str(new_name))
        if updated == content:
            continue
        file_path.write_text(updated, encoding="utf-8")
        updated_paths.append(Path(relative_path).as_posix())

    return {
        "success": bool(updated_paths),
        "controller": skill.controller,
        "old_name": old_name,
        "new_name": new_name,
        "updated_paths": updated_paths,
        "message": f"Updated {len(updated_paths)} file(s).",
    }


register_controller("mock_annotate_formatted", _mock_annotate_formatted)
register_controller("mock_create_pytest_skeleton", _mock_create_pytest_skeleton)
register_controller("mock_toggle_to_passing", _mock_toggle_to_passing)
register_controller("mock_update_imports", _mock_update_imports)


def _format_on_save() -> Skill:
    """Trigger: glob ``*.py``.  Controller: mock annotation update."""
    return Skill(
        id="format_on_save",
        name="Format on Save",
        triggers=(Trigger(kind="glob", spec="*.py", salience=1.0),),
        salience=1.0,
        prior=0.9,
        signature=TypedSignature(
            input_type="file_path",
            output_type="annotation_update",
            effects=frozenset({Effect.WRITE_FS}),
        ),
        controller="mock_annotate_formatted",
        cost_estimate=0.01,
        preconditions=(),
        provenance="sample_skills",
    )


def _add_test_for_new_function(workspace: Workspace) -> Skill:
    """Trigger: annotation tags ``needs_test`` AND ``py``.

    Controller: creates a minimal pytest skeleton annotation.
    """
    return Skill(
        id="add_test_for_new_function",
        name="Add Test for New Function",
        triggers=(
            Trigger(kind="annotation", spec="needs_test", salience=0.9),
            Trigger(kind="annotation", spec="py", salience=0.5),
        ),
        salience=0.8,
        prior=0.7,
        signature=TypedSignature(
            input_type="annotation",
            output_type="test_file",
            effects=frozenset({Effect.WRITE_FS}),
        ),
        controller="mock_create_pytest_skeleton",
        cost_estimate=0.05,
        preconditions=("needs_test",),
        provenance="sample_skills",
    )


def _fix_failing_test() -> Skill:
    """Trigger: annotation status ``failing``.

    Controller: toggles annotation to ``passing``.
    """
    return Skill(
        id="fix_failing_test",
        name="Fix Failing Test",
        triggers=(Trigger(kind="annotation", spec="failing", salience=1.0),),
        salience=1.0,
        prior=0.6,
        signature=TypedSignature(
            input_type="annotation_status",
            output_type="annotation_update",
            effects=frozenset({Effect.WRITE_FS, Effect.RUN_SHELL}),
        ),
        controller="mock_toggle_to_passing",
        cost_estimate=0.1,
        preconditions=(),
        provenance="sample_skills",
    )


def _update_imports_after_rename() -> Skill:
    """Trigger: regex matching a rename event.

    Controller: deterministic string-level import update.
    """
    return Skill(
        id="update_imports_after_rename",
        name="Update Imports After Rename",
        triggers=(
            Trigger(kind="regex", spec=".*rename.*", salience=0.8),
        ),
        salience=0.7,
        prior=0.8,
        signature=TypedSignature(
            input_type="rename_event",
            output_type="file_update",
            effects=frozenset({Effect.WRITE_FS}),
        ),
        controller="mock_update_imports",
        cost_estimate=0.03,
        preconditions=(),
        provenance="sample_skills",
    )


def _escalate_when_stuck() -> Skill:
    """Represented only through the dispatcher fallback — not a real skill.

    Returned here for completeness so the demo can list it.
    """
    return Skill(
        id="escalate_when_stuck",
        name="Escalate When Stuck",
        triggers=(),
        salience=0.0,
        prior=0.0,
        controller="fallback_planner",
        cost_estimate=0.0,
        preconditions=(),
        provenance="sample_skills",
    )


def create_sample_skills(workspace: Workspace) -> list[Skill]:
    """Create and return all sample skills.

    Args:
        workspace: The workspace instance (used by skills that inspect
            annotations during construction).

    Returns:
        A list of five :class:`Skill` instances.
    """
    return [
        _format_on_save(),
        _add_test_for_new_function(workspace),
        _fix_failing_test(),
        _update_imports_after_rename(),
        _escalate_when_stuck(),
    ]
