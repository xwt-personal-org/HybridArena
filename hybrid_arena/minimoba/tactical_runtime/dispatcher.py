"""Deterministic tactical skill dispatch for MiniMOBA."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

TacticalAction = dict[str, int]
Controller = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class TacticalSkill:
    skill_id: str
    score: float
    controller: str


@dataclass(frozen=True)
class TacticalDispatchResult:
    skill_id: str | None
    action: TacticalAction | None
    escalated: bool
    success: bool
    residual: float
    message: str


_CONTROLLER_REGISTRY: dict[str, Controller] = {
    "noop": lambda state: {"move": 0, "skill": 0, "target": 0},
}


class TacticalDispatcher:
    def __init__(self, controllers: dict[str, Controller] | None = None):
        self.controllers = dict(_CONTROLLER_REGISTRY)
        if controllers:
            self.controllers.update(controllers)
        self.trace: list[dict[str, Any]] = []

    def dispatch(
        self,
        skills: list[TacticalSkill],
        state: Mapping[str, Any],
    ) -> TacticalDispatchResult:
        if not skills:
            result = TacticalDispatchResult(
                skill_id=None,
                action=None,
                escalated=False,
                success=False,
                residual=1.0,
                message="No tactical skill candidates",
            )
            self.trace.append({"success": False, "error": "no_candidates"})
            return result

        winner = max(skills, key=lambda skill: skill.score)
        controller = self.controllers.get(winner.controller)
        if controller is None:
            result = TacticalDispatchResult(
                skill_id=winner.skill_id,
                action=None,
                escalated=False,
                success=False,
                residual=1.0,
                message=f"Unknown controller: {winner.controller}",
            )
            self.trace.append(
                {
                    "skill_id": winner.skill_id,
                    "controller": winner.controller,
                    "success": False,
                    "error": "unknown_controller",
                }
            )
            return result

        action = controller(state)
        if not _is_valid_action(action):
            result = TacticalDispatchResult(
                skill_id=winner.skill_id,
                action=None,
                escalated=False,
                success=False,
                residual=1.0,
                message=f"Invalid controller action: {winner.controller}",
            )
            self.trace.append(
                {
                    "skill_id": winner.skill_id,
                    "controller": winner.controller,
                    "success": False,
                    "error": "invalid_controller_action",
                    "action": action,
                }
            )
            return result

        result = TacticalDispatchResult(
            skill_id=winner.skill_id,
            action=dict(action),
            escalated=False,
            success=True,
            residual=max(0.0, 1.0 - float(winner.score)),
            message=f"Dispatched controller: {winner.controller}",
        )
        self.trace.append(
            {
                "skill_id": winner.skill_id,
                "controller": winner.controller,
                "success": True,
                "action": dict(action),
            }
        )
        return result


def _is_valid_action(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    if set(action) != {"move", "skill", "target"}:
        return False
    return all(type(value) is int for value in action.values())
