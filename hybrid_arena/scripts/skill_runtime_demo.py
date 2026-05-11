"""CLI demo for the AgentBench skill-runtime L0/L1 prototype.

Usage::

    python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_arena.skill_runtime.body_schema import BodySchema
from hybrid_arena.skill_runtime.dispatcher import ReflexDispatcher
from hybrid_arena.skill_runtime.sample_skills import create_sample_skills
from hybrid_arena.skill_runtime.schema import WorkspaceEvent
from hybrid_arena.skill_runtime.workspace import Workspace


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the skill-runtime demo CLI."""
    parser = argparse.ArgumentParser(
        description="Run a single skill-runtime dispatch cycle."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--db",
        default=".skills/state.db",
        help="Path to the SQLite state database (default: .skills/state.db).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one synthetic dispatch and exit.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = root / db_path

    # Bootstrap workspace, skills, body-schema and dispatcher.
    workspace = Workspace(root=root, db_path=db_path)
    skills = create_sample_skills(workspace)
    body = BodySchema(skills=skills, workspace=workspace)
    dispatcher = ReflexDispatcher(body=body, workspace=workspace)

    # Synthetic event: a Python file was saved.
    event = WorkspaceEvent(kind="file_save", path="hybrid_arena/core/utils.py")

    result = dispatcher.dispatch(event)

    # Report.
    if result.skill_id:
        print(f"Selected skill: {result.skill_id}")
    else:
        print("No skill selected (escalated).")

    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    trace_count = len(workspace.get_traces())
    print(f"Trace count: {trace_count}")

    if not args.once:
        print("\n(Use --once to run a single cycle and exit.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
