"""Deterministic sample skills used by API and CLI smoke paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_arena.skill_runtime.schema import Effect, Skill, SkillSignature, WorkspaceEvent


def build_sample_skills() -> list[Skill]:
    return [
        Skill(
            id="inspect_workspace",
            name="Inspect workspace path",
            description="Read-only inspection of a path inside the configured runtime workspace.",
            signature=SkillSignature(
                inputs={"path": "str"},
                outputs={"exists": "bool", "path": "str"},
                effects=(Effect.READ_FS,),
            ),
        ),
        Skill(
            id="write_summary",
            name="Write deterministic summary",
            description="Write a deterministic Markdown summary for local demo runs.",
            signature=SkillSignature(
                inputs={"path": "str"},
                outputs={"path": "str"},
                effects=(Effect.READ_FS, Effect.WRITE_FS),
            ),
        ),
        Skill(
            id="run_shell",
            name="Run shell command",
            description="Placeholder shell-capable tool; never auto-enabled in API defaults.",
            signature=SkillSignature(
                inputs={"command": "str"},
                outputs={"stdout": "str"},
                effects=(Effect.RUN_SHELL,),
            ),
        ),
        Skill(
            id="fetch_url",
            name="Fetch URL",
            description="Placeholder network-capable tool; never auto-enabled in API defaults.",
            signature=SkillSignature(
                inputs={"url": "str"},
                outputs={"status": "int"},
                effects=(Effect.NETWORK,),
            ),
        ),
        Skill(
            id="ask_llm",
            name="Ask LLM",
            description="Placeholder LLM-capable tool; no real LLM calls are implemented.",
            signature=SkillSignature(
                inputs={"prompt": "str"},
                outputs={"text": "str"},
                effects=(Effect.LLM_CALL,),
            ),
        ),
    ]


def choose_sample_skill(event: WorkspaceEvent, skills: list[Skill]) -> Skill | None:
    requested = event.payload.get("action") or event.payload.get("tool_id")
    if requested is None:
        requested = "inspect_workspace"
    for skill in skills:
        if skill.id == requested:
            return skill
    return None


def execute_sample_skill(skill: Skill, event: WorkspaceEvent, workspace_root: Path) -> dict[str, Any]:
    path = _resolve_workspace_path(workspace_root, event.path)
    if skill.id == "inspect_workspace":
        return {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }
    if skill.id == "write_summary":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Skill Runtime Summary\n\n"
            f"- event_kind: {event.kind}\n"
            f"- event_path: {event.path}\n",
            encoding="utf-8",
        )
        return {"path": str(path), "bytes": path.stat().st_size}
    if skill.id == "run_shell":
        return {"stdout": "shell execution is disabled in this deterministic demo"}
    if skill.id == "fetch_url":
        return {"status": 0, "message": "network access is disabled in this deterministic demo"}
    if skill.id == "ask_llm":
        return {"text": "LLM calls are not implemented in this deterministic demo"}
    raise ValueError(f"Unknown sample skill: {skill.id}")


def _resolve_workspace_path(workspace_root: Path, event_path: str) -> Path:
    root = workspace_root.resolve()
    candidate = (root / event_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes skill-runtime workspace: {event_path}")
    return candidate

