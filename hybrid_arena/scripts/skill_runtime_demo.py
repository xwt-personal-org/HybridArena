"""CLI demo for the AgentBench skill-runtime L0/L1 prototype.

Usage::

    python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_arena.skill_runtime.body_schema import BodySchema
from hybrid_arena.skill_runtime.dispatcher import ReflexDispatcher
from hybrid_arena.skill_runtime.protocol import EnvelopeKind, SkillRuntimeMessage
from hybrid_arena.skill_runtime.sample_skills import create_sample_skills
from hybrid_arena.skill_runtime.schema import WorkspaceEvent
from hybrid_arena.skill_runtime.tool_registry import ToolRegistry
from hybrid_arena.skill_runtime.workspace import Workspace


def _load_event(path: Path | None) -> WorkspaceEvent:
    if path is None:
        return WorkspaceEvent(kind="file_save", path="hybrid_arena/core/utils.py")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") or (raw.get("kind") == EnvelopeKind.EVENT.value and "payload" in raw):
        return SkillRuntimeMessage.from_dict(raw).to_workspace_event()
    return WorkspaceEvent(
        kind=str(raw["kind"]),
        path=str(raw.get("path", "")),
        payload=dict(raw.get("payload", {})),
        created_at=float(raw.get("created_at", 0.0)),
    )


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
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List deterministic built-in controller descriptors as JSON.",
    )
    parser.add_argument(
        "--explain-affordances",
        action="store_true",
        help="Print body-schema affordance diagnostics as JSON.",
    )
    parser.add_argument(
        "--event-json",
        type=Path,
        help="Path to a JSON WorkspaceEvent or event envelope.",
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
    event = _load_event(args.event_json)

    if args.list_tools:
        registry = ToolRegistry.discover_builtin_controllers()
        print(
            json.dumps(
                {"tools": [tool.to_dict() for tool in registry.list()]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.explain_affordances:
        print(
            json.dumps(
                body.explain_affordances(event=event),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

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
