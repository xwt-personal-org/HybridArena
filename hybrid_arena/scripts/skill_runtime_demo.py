"""CLI diagnostics for the deterministic skill runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_arena.skill_runtime.adviser import SkillRuntimeAdviser
from hybrid_arena.skill_runtime.memory import SkillMemoryStore
from hybrid_arena.skill_runtime.sample_skills import build_sample_skills
from hybrid_arena.skill_runtime.security import RuntimePermissionPolicy
from hybrid_arena.skill_runtime.tool_registry import list_tools, summarize_policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic skill-runtime diagnostics")
    parser.add_argument("--root", default=".", help="Skill runtime workspace root")
    parser.add_argument("--db", default=".skills/state.db", help="Skill runtime SQLite state path")
    parser.add_argument("--once", action="store_true", help="Run one diagnostic pass")
    parser.add_argument("--list-tools", action="store_true", help="List sample tools and policy status")
    parser.add_argument(
        "--explain-affordances",
        action="store_true",
        help="Print deterministic policy and memory advisories",
    )
    parser.add_argument("--policy-summary", action="store_true", help="Print policy summary counts")
    parser.add_argument(
        "--allow-write-effects",
        action="store_true",
        help="Enable write-capable demo tools for local development",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    policy = RuntimePermissionPolicy(
        allow_write_effects=args.allow_write_effects,
        workspace_root=root.resolve(),
    )
    skills = build_sample_skills()
    store = SkillMemoryStore(args.db)

    if args.list_tools:
        for tool in list_tools(policy, skills):
            reason = f" reason={tool['blocked_reason']}" if tool["blocked_reason"] else ""
            effects = ",".join(tool["effects"])
            print(f"{tool['id']} allowed={tool['allowed']} effects={effects}{reason}")

    if args.explain_affordances:
        adviser = SkillRuntimeAdviser(store)
        for item in adviser.advise(skills=skills, policy=policy):
            print(f"{item['id']}: {item['message']}")

    if args.policy_summary:
        summary = summarize_policy(policy, skills)
        print(f"allowed_tools: {len(summary['allowed_tools'])}")
        print(f"blocked_tools: {len(summary['blocked_tools'])}")
        print(f"blocked_effects: {', '.join(summary['blocked_effects'])}")

    if not (args.list_tools or args.explain_affordances or args.policy_summary):
        print("skill-runtime demo ready")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

