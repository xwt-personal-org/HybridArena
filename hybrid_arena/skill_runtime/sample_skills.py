"""Deterministic sample skills for the L0/L1 prototype.

No LLM or external API calls — every skill is a pure, deterministic mock
that exercises the dispatch→trace pipeline.
"""

from __future__ import annotations

from hybrid_arena.skill_runtime.schema import (
    Effect,
    Skill,
    Trigger,
    TypedSignature,
)
from hybrid_arena.skill_runtime.workspace import Workspace


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
