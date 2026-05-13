"""Deterministic advisories for the skill runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from hybrid_arena.skill_runtime.memory import SkillMemoryStore
from hybrid_arena.skill_runtime.schema import Skill
from hybrid_arena.skill_runtime.workspace import Workspace


@dataclass(frozen=True, init=False)
class Advisory:
    """One local runtime advisory."""

    code: str
    severity: str
    message: str
    suggested_action: str
    evidence: dict[str, Any]

    def __init__(
        self,
        *,
        code: str = "",
        severity: str,
        message: str,
        suggested_action: str = "",
        evidence: dict[str, Any] | None = None,
        kind: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "code", code or kind)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "suggested_action", suggested_action)
        object.__setattr__(self, "evidence", dict(evidence if evidence is not None else metadata or {}))

    @property
    def kind(self) -> str:
        """Compatibility alias for ``code``."""
        return self.code

    @property
    def metadata(self) -> dict[str, Any]:
        """Compatibility alias for ``evidence``."""
        return self.evidence

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable advisory."""
        return {
            "code": self.code,
            "kind": self.code,
            "severity": self.severity,
            "message": self.message,
            "suggested_action": self.suggested_action,
            "evidence": self.evidence,
            "metadata": self.evidence,
        }


class SkillRuntimeAdviser:
    """Analyze local runtime state and emit deterministic advisories."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        memory: SkillMemoryStore | None = None,
    ) -> None:
        self._workspace = workspace
        self._memory = memory

    def advise(self, *, now: float | None = None) -> list[Advisory]:
        """Return advisories for current workspace and trace state."""
        if self._workspace is None:
            return []
        advisories: list[Advisory] = []
        advisories.extend(self.evaluate_empty_workspace(self._workspace, skills=[]))
        advisories.extend(self.evaluate_escalation_rate(self._workspace))
        advisories.extend(self.evaluate_low_success_skills(self._workspace, self._memory))
        advisories.extend(self.evaluate_stale_annotations(self._workspace, now=now))
        return advisories

    def evaluate_empty_workspace(
        self,
        workspace: Workspace,
        skills: list[Skill],
    ) -> list[Advisory]:
        """Report empty state or missing skills/tools."""
        annotations = workspace.snapshot_annotations()
        traces = workspace.get_traces(limit=1)
        advisories: list[Advisory] = []
        if not annotations and not traces:
            advisories.append(
                Advisory(
                    code="empty_workspace",
                    severity="info",
                    message="No annotations or traces are available yet.",
                    suggested_action="Run a deterministic dispatch or seed workspace annotations.",
                    evidence={"annotation_count": 0, "trace_count": 0},
                )
            )
        if not skills:
            advisories.append(
                Advisory(
                    code="no_registered_skills",
                    severity="info",
                    message="No skills were supplied to adviser evaluation.",
                    suggested_action="Register deterministic sample skills before dispatch.",
                    evidence={"skill_count": 0},
                )
            )
        return advisories

    def evaluate_escalation_rate(
        self,
        workspace: Workspace,
        threshold: float = 0.5,
    ) -> list[Advisory]:
        """Report repeated escalation in recent traces."""
        traces = workspace.get_traces(limit=200)
        if not traces:
            return []
        escalation_count = sum(
            1
            for trace in traces
            if trace["skill_id"] in {"__escalated__", "fallback"}
            or trace["output_snapshot"].get("escalated") is True
        )
        rate = escalation_count / len(traces)
        if escalation_count >= 3 and rate >= threshold:
            return [
                Advisory(
                    code="repeated_escalation",
                    severity="warning",
                    message="Recent dispatches escalated repeatedly.",
                    suggested_action="Add path-specific annotations or adjust trigger coverage.",
                    evidence={"count": escalation_count, "rate": rate, "threshold": threshold},
                )
            ]
        return []

    def evaluate_low_success_skills(
        self,
        workspace: Workspace,
        memory: SkillMemoryStore | None = None,
    ) -> list[Advisory]:
        """Report skills with low recent trace success."""
        traces = workspace.get_traces(limit=200)
        by_skill: dict[str, list[dict[str, Any]]] = {}
        for trace in traces:
            skill_id = trace["skill_id"]
            if skill_id.startswith("__"):
                continue
            by_skill.setdefault(skill_id, []).append(trace)

        advisories: list[Advisory] = []
        for skill_id, skill_traces in sorted(by_skill.items()):
            if len(skill_traces) < 2:
                continue
            successes = sum(1 for trace in skill_traces if trace["success"])
            success_rate = successes / len(skill_traces)
            if success_rate < 0.6:
                advisories.append(
                    Advisory(
                        code="low_success_skill",
                        severity="warning",
                        message=f"Skill {skill_id} has low recent success.",
                        suggested_action="Inspect traces and increase no-go penalty or repair coverage.",
                        evidence={
                            "skill_id": skill_id,
                            "success_rate": success_rate,
                            "total": len(skill_traces),
                            "memory_attached": memory is not None,
                        },
                    )
                )
        return advisories

    def evaluate_stale_annotations(
        self,
        workspace: Workspace,
        now: float | None = None,
    ) -> list[Advisory]:
        """Report annotations whose decay timestamp has elapsed."""
        current = time.time() if now is None else now
        stale_paths = [
            item.path
            for item in workspace.snapshot_annotations()
            if item.decay_at > 0.0 and item.decay_at <= current
        ]
        if not stale_paths:
            return []
        return [
            Advisory(
                code="stale_annotations",
                severity="warning",
                message="Some annotations have expired and should be pruned.",
                suggested_action="Call Workspace.prune_expired_annotations().",
                evidence={"paths": stale_paths},
            )
        ]
