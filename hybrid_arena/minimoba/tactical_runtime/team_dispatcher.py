"""Team-level tactical dispatcher with deterministic conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

from hybrid_arena.minimoba.tactical_runtime.body_schema import GameBodySchema
from hybrid_arena.minimoba.tactical_runtime.dispatcher import (
    TacticalDispatcher,
    TacticalDispatchResult,
)
from hybrid_arena.minimoba.tactical_runtime.memory import TacticalMemoryStore
from hybrid_arena.minimoba.tactical_runtime.schema import GameSkill
from hybrid_arena.minimoba.tactical_runtime.skills import nearest_tagged_region
from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace, GameEvent

FALLBACK_PATROL_ACTION = {"move": 0, "skill": 0, "target": 0}


@dataclass
class TeamDispatchResult:
    """Actions, selected skills, conflicts, and trace diagnostics."""

    actions: dict[str, dict[str, int]]
    selected_skills: dict[str, str | None]
    conflicts: list[dict[str, object]] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)
    agent_results: dict[str, TacticalDispatchResult] = field(default_factory=dict)


class TeamTacticalDispatcher:
    """Dispatch tactical actions for a team and reroute resource conflicts."""

    def __init__(
        self,
        skills: list[GameSkill] | None = None,
        workspace: BattlefieldWorkspace | None = None,
        fallback_planner: object = None,
        memory: TacticalMemoryStore | None = None,
        *,
        body: GameBodySchema | None = None,
    ) -> None:
        if workspace is None and body is not None:
            workspace = body.workspace
        if workspace is None:
            raise ValueError("workspace is required")
        self.workspace = workspace
        self.memory = memory
        self.body = body or GameBodySchema(skills=skills or [], workspace=workspace)
        self.dispatcher = TacticalDispatcher(
            body=self.body,
            workspace=workspace,
            fallback_planner=fallback_planner,
        )

    def dispatch_team(
        self,
        events_by_agent: dict[str, GameEvent] | None = None,
        game_state: object = None,
        agent_ids: list[str] | tuple[str, ...] | None = None,
        *,
        events: dict[str, GameEvent] | None = None,
    ) -> TeamDispatchResult:
        """Dispatch all requested agents and resolve shared target conflicts."""
        events_map = events_by_agent if events_by_agent is not None else events or {}
        ordered_agents = list(agent_ids) if agent_ids is not None else sorted(events_map)
        agent_results: dict[str, TacticalDispatchResult] = {}
        actions: dict[str, dict[str, int]] = {}
        selected_skills: dict[str, str | None] = {}
        selected_targets: dict[str, tuple[int, int]] = {}
        trace: list[dict[str, object]] = []

        for agent_id in ordered_agents:
            event = events_map.get(agent_id, GameEvent(kind="tick", agent_id=agent_id))
            result = self.dispatcher.dispatch(event, game_state=game_state, agent_id=agent_id)
            agent_results[agent_id] = result
            actions[agent_id] = self._normalize_action(result.action)
            selected_skills[agent_id] = result.skill_id
            target = self._selected_objective_or_resource(result, game_state, agent_id)
            if target is not None:
                selected_targets[agent_id] = target
            trace.append({
                "agent_id": agent_id,
                "event_kind": event.kind,
                "skill_id": result.skill_id,
                "action": dict(actions[agent_id]),
                "target": target,
            })

        conflicts = self._resolve_conflicts(actions, selected_targets, game_state)
        return TeamDispatchResult(
            actions=actions,
            selected_skills=selected_skills,
            conflicts=conflicts,
            trace=trace,
            agent_results=agent_results,
        )

    def _resolve_conflicts(
        self,
        actions: dict[str, dict[str, int]],
        selected_targets: dict[str, tuple[int, int]],
        game_state: object,
    ) -> list[dict[str, object]]:
        by_target: dict[tuple[int, int], list[str]] = {}
        for agent_id, target in selected_targets.items():
            by_target.setdefault(target, []).append(agent_id)

        conflicts: list[dict[str, object]] = []
        for target in sorted(by_target):
            agents = sorted(by_target[target])
            if len(agents) <= 1:
                continue

            kept_agent = min(
                agents,
                key=lambda agent_id: (
                    self._distance_to_target(agent_id, target, game_state),
                    agent_id,
                ),
            )
            rerouted_agents = [agent_id for agent_id in agents if agent_id != kept_agent]
            for agent_id in rerouted_agents:
                actions[agent_id] = dict(FALLBACK_PATROL_ACTION)
            conflicts.append({
                "target": target,
                "kept_agent": kept_agent,
                "rerouted_agents": rerouted_agents,
            })
        return conflicts

    def _selected_objective_or_resource(
        self,
        result: TacticalDispatchResult,
        game_state: object,
        agent_id: str,
    ) -> tuple[int, int] | None:
        position = self._agent_position(agent_id, game_state)
        if result.skill_id == "farm_resources":
            target = nearest_tagged_region(self.workspace, position, "resource_soon")
            if target is not None:
                return target
            return nearest_tagged_region(self.workspace, position, "objective")
        if result.skill_id == "push_objective":
            return nearest_tagged_region(self.workspace, position, "objective")
        return None

    def _distance_to_target(
        self,
        agent_id: str,
        target: tuple[int, int],
        game_state: object,
    ) -> int:
        ax, ay = self._agent_position(agent_id, game_state)
        tx, ty = target
        return max(abs(tx - ax), abs(ty - ay))

    def _agent_position(self, agent_id: str, game_state: object) -> tuple[int, int]:
        hero = getattr(game_state, "heroes", {}).get(agent_id) if game_state else None
        if hero is None:
            return (0, 0)
        return int(getattr(hero, "x", 0)), int(getattr(hero, "y", 0))

    def _normalize_action(self, action: dict | None) -> dict[str, int]:
        if action is None:
            return dict(FALLBACK_PATROL_ACTION)
        return {
            "move": int(action.get("move", 0)),
            "skill": int(action.get("skill", 0)),
            "target": int(action.get("target", 0)),
        }
