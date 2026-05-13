"""Deterministic skill-runtime dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybrid_arena.skill_runtime.sample_skills import choose_sample_skill, execute_sample_skill
from hybrid_arena.skill_runtime.schema import Skill, WorkspaceEvent
from hybrid_arena.skill_runtime.security import (
    RuntimePermissionPolicy,
    SkillRuntimePermissionError,
    blocked_reason,
    is_skill_allowed,
)


@dataclass(frozen=True)
class SkillDispatchResult:
    tool_id: str
    success: bool
    output: dict[str, Any]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "success": self.success,
            "output": self.output,
            "message": self.message,
        }


def dispatch_workspace_event(
    event: WorkspaceEvent,
    skills: list[Skill],
    policy: RuntimePermissionPolicy,
    workspace_root: Path,
) -> SkillDispatchResult | None:
    skill = choose_sample_skill(event, skills)
    if skill is None:
        return None
    if not is_skill_allowed(skill, policy):
        raise SkillRuntimePermissionError(skill, blocked_reason(skill, policy))
    output = execute_sample_skill(skill, event, workspace_root)
    return SkillDispatchResult(
        tool_id=skill.id,
        success=True,
        output=output,
        message=f"Dispatched {skill.id}",
    )

