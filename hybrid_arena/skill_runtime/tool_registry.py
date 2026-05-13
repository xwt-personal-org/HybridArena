"""Tool registry helpers with policy diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_arena.skill_runtime.sample_skills import build_sample_skills
from hybrid_arena.skill_runtime.schema import Effect, Skill
from hybrid_arena.skill_runtime.security import (
    RuntimePermissionPolicy,
    blocked_reason,
    denied_effects,
    is_skill_allowed,
    skill_effects,
)


def tool_metadata(skill: Skill, policy: RuntimePermissionPolicy) -> dict[str, Any]:
    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "effects": sorted(effect.value for effect in skill_effects(skill)),
        "allowed": is_skill_allowed(skill, policy),
        "blocked_reason": blocked_reason(skill, policy),
    }


def list_tools(
    policy: RuntimePermissionPolicy | None = None,
    skills: list[Skill] | None = None,
) -> list[dict[str, Any]]:
    active_policy = policy or RuntimePermissionPolicy()
    return [tool_metadata(skill, active_policy) for skill in skills or build_sample_skills()]


def summarize_policy(
    policy: RuntimePermissionPolicy,
    skills: list[Skill] | None = None,
) -> dict[str, Any]:
    active_skills = skills or build_sample_skills()
    tools = list_tools(policy, active_skills)
    allowed_tools = [tool["id"] for tool in tools if tool["allowed"]]
    blocked_tools = [tool for tool in tools if not tool["allowed"]]
    blocked_effects: set[str] = set()
    skill_by_id = {skill.id: skill for skill in active_skills}
    for tool in blocked_tools:
        blocked_effects.update(effect.value for effect in denied_effects(skill_by_id[tool["id"]], policy))
    return {
        "allow_write_effects": policy.allow_write_effects,
        "workspace_root": str(Path(policy.workspace_root).resolve()) if policy.workspace_root else None,
        "allowed_tools": sorted(allowed_tools),
        "blocked_tools": sorted(
            (
                {"id": tool["id"], "effects": tool["effects"], "reason": tool["blocked_reason"]}
                for tool in blocked_tools
            ),
            key=lambda item: item["id"],
        ),
        "blocked_effects": sorted(blocked_effects),
    }


def effect_from_name(name: str) -> Effect:
    try:
        return Effect[name]
    except KeyError as exc:
        raise ValueError(f"Unknown effect: {name}") from exc
