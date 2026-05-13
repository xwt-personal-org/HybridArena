"""Permission policy for deterministic skill-runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hybrid_arena.skill_runtime.schema import Effect, Skill

DEFAULT_BLOCKED_EFFECTS = frozenset(
    {Effect.WRITE_FS, Effect.RUN_SHELL, Effect.NETWORK, Effect.LLM_CALL}
)


class SkillRuntimePermissionError(Exception):
    """Raised when policy blocks a skill invocation."""

    def __init__(self, skill: Skill, reason: str):
        super().__init__(reason)
        self.skill = skill
        self.reason = reason


@dataclass(frozen=True)
class RuntimePermissionPolicy:
    allow_write_effects: bool = False
    allowed_effects: frozenset[Effect] | None = None
    workspace_root: Path | None = None


def skill_effects(skill: Skill) -> frozenset[Effect]:
    return frozenset(skill.signature.effects)


def denied_effects(skill: Skill, policy: RuntimePermissionPolicy) -> frozenset[Effect]:
    if policy.allow_write_effects:
        return frozenset()
    explicitly_allowed = policy.allowed_effects or frozenset()
    return (skill_effects(skill) & DEFAULT_BLOCKED_EFFECTS) - explicitly_allowed


def is_skill_allowed(skill: Skill, policy: RuntimePermissionPolicy) -> bool:
    return not denied_effects(skill, policy)


def filter_allowed_skills(skills: list[Skill], policy: RuntimePermissionPolicy) -> list[Skill]:
    return [skill for skill in skills if is_skill_allowed(skill, policy)]


def blocked_reason(skill: Skill, policy: RuntimePermissionPolicy) -> str:
    blocked = denied_effects(skill, policy)
    if not blocked:
        return ""
    effects = ", ".join(sorted(effect.value for effect in blocked))
    return f"Blocked effects: {effects}"

