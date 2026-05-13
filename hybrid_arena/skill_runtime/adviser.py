"""Deterministic policy and memory advisories for the skill runtime."""

from __future__ import annotations

import time
from typing import Any

from hybrid_arena.skill_runtime.memory import SkillMemoryStore
from hybrid_arena.skill_runtime.schema import Effect, Skill
from hybrid_arena.skill_runtime.security import (
    RuntimePermissionPolicy,
    is_skill_allowed,
    skill_effects,
)
from hybrid_arena.skill_runtime.tool_registry import summarize_policy


class SkillRuntimeAdviser:
    def __init__(self, memory_store: SkillMemoryStore):
        self.memory_store = memory_store

    def advise(
        self,
        skills: list[Skill],
        policy: RuntimePermissionPolicy,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        timestamp = time.time() if now is None else now
        advice: list[dict[str, Any]] = []
        policy_summary = summarize_policy(policy, skills)

        advice.append(
            {
                "id": "tool_policy_summary",
                "severity": "info",
                "allowed_tool_count": len(policy_summary["allowed_tools"]),
                "blocked_tool_count": len(policy_summary["blocked_tools"]),
                "blocked_effects": policy_summary["blocked_effects"],
                "message": "Tool policy is active for skill-runtime dispatch.",
            }
        )

        write_capable_blocked = [
            skill
            for skill in skills
            if Effect.WRITE_FS in skill_effects(skill) and not is_skill_allowed(skill, policy)
        ]
        if write_capable_blocked:
            advice.append(
                {
                    "id": "write_effects_disabled",
                    "severity": "warning",
                    "blocked_tools": sorted(skill.id for skill in write_capable_blocked),
                    "message": "Write-capable tools are blocked unless explicit dev/test opt-in is enabled.",
                }
            )

        stale_records = self.memory_store.stale_records(now=timestamp)
        low_success = self.memory_store.low_success_trace_summaries()
        if stale_records or low_success:
            advice.append(
                {
                    "id": "memory_hygiene",
                    "severity": "warning",
                    "stale_record_count": len(stale_records),
                    "low_success_skill_count": len(low_success),
                    "low_success_skills": [summary.skill_id for summary in low_success],
                    "message": "Memory store has stale annotations or low-success trace summaries.",
                }
            )

        return advice

