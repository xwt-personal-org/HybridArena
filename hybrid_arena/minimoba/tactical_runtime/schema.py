"""Schema types for the tactical runtime: effects, triggers, skills, forward models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace


class GameEffect(Enum):
    """Types of game effects that skills can produce."""

    MOVE = "move"
    ATTACK = "attack"
    USE_ABILITY = "use_ability"
    RETREAT = "retreat"
    LLM_CALL = "llm_call"


@dataclass
class GameTrigger:
    """A trigger condition that evaluates to a score for skill activation.

    Supported kinds:
        - health_threshold: spec "below:0.25" or "above:0.75"
        - enemy_count: spec "nearby:3" (within radius)
        - team_state: spec "advantage" or "disadvantage"
        - annotation_query: spec "any:tag1,tag2" or "all:tag1,tag2"
    """

    kind: str
    spec: str
    salience: float = 1.0
    evaluate: object = None  # (workspace, game_state, agent_id) -> float

    def score(
        self,
        workspace: BattlefieldWorkspace,
        game_state: object = None,
        agent_id: str = "",
    ) -> float:
        """Evaluate this trigger and return a score in [0, 1].

        Args:
            workspace: The battlefield workspace for annotation queries.
            game_state: Optional game state for health/enemy queries.
            agent_id: The agent being evaluated.

        Returns:
            Score in [0, 1]. 0 means trigger does not fire.
        """
        if self.evaluate is not None:
            return float(self.evaluate(workspace, game_state, agent_id)) * self.salience

        if self.kind == "health_threshold":
            return self._eval_health_threshold(game_state, agent_id) * self.salience
        if self.kind == "enemy_count":
            return self._eval_enemy_count(workspace, game_state, agent_id) * self.salience
        if self.kind == "team_state":
            return self._eval_team_state(game_state, agent_id) * self.salience
        if self.kind == "annotation_query":
            return self._eval_annotation_query(workspace, agent_id) * self.salience
        return 0.0

    def _eval_health_threshold(
        self, game_state: object, agent_id: str
    ) -> float:
        """Evaluate health_threshold trigger.

        spec format: "below:0.25" or "above:0.75"
        """
        if game_state is None or not agent_id:
            return 0.0
        hero = getattr(game_state, "heroes", {}).get(agent_id)
        if hero is None:
            return 0.0
        hp_ratio = getattr(hero, "hp", 0) / max(getattr(hero, "config", None).max_hp, 1)
        match = re.match(r"(below|above):([\d.]+)", self.spec)
        if not match:
            return 0.0
        direction, threshold = match.group(1), float(match.group(2))
        if direction == "below" and hp_ratio < threshold:
            return 1.0 - hp_ratio  # lower health = higher urgency
        if direction == "above" and hp_ratio > threshold:
            return hp_ratio
        return 0.0

    def _eval_enemy_count(
        self, workspace: BattlefieldWorkspace, game_state: object, agent_id: str
    ) -> float:
        """Evaluate enemy_count trigger.

        spec format: "nearby:3" (within radius)
        """
        if game_state is None:
            return 0.0
        match = re.match(r"nearby:(\d+)", self.spec)
        if not match:
            return 0.0
        radius = int(match.group(1))
        hero = getattr(game_state, "heroes", {}).get(agent_id)
        if hero is None:
            return 0.0
        hx, hy = getattr(hero, "x", 0), getattr(hero, "y", 0)
        team = getattr(hero, "team", "")
        count = 0
        for other_id, other_hero in getattr(game_state, "heroes", {}).items():
            if other_id == agent_id or not getattr(other_hero, "alive", False):
                continue
            if getattr(other_hero, "team", "") == team:
                continue
            ox, oy = getattr(other_hero, "x", 0), getattr(other_hero, "y", 0)
            if max(abs(ox - hx), abs(oy - hy)) <= radius:
                count += 1
        if count >= 1:
            return min(count / max(radius, 1), 1.0)
        return 0.0

    def _eval_team_state(self, game_state: object, agent_id: str) -> float:
        """Evaluate team_state trigger.

        spec format: "advantage" or "disadvantage"
        """
        if game_state is None or not agent_id:
            return 0.0
        hero = getattr(game_state, "heroes", {}).get(agent_id)
        if hero is None:
            return 0.0
        team = getattr(hero, "team", "")
        red_kills = getattr(game_state, "red_kills", 0)
        blue_kills = getattr(game_state, "blue_kills", 0)
        red_towers = getattr(game_state, "red_towers", 2)
        blue_towers = getattr(game_state, "blue_towers", 2)
        if team == "red":
            advantage = (red_kills - blue_kills) + (red_towers - blue_towers) * 2
        elif team == "blue":
            advantage = (blue_kills - red_kills) + (blue_towers - red_towers) * 2
        else:
            return 0.0
        if self.spec == "advantage" and advantage > 0:
            return min(abs(advantage) / 10.0, 1.0)
        if self.spec == "disadvantage" and advantage < 0:
            return min(abs(advantage) / 10.0, 1.0)
        return 0.0

    def _eval_annotation_query(
        self, workspace: BattlefieldWorkspace, agent_id: str
    ) -> float:
        """Evaluate annotation_query trigger.

        spec format: "any:tag1,tag2" or "all:tag1,tag2"
        """
        match = re.match(r"(any|all):(.+)", self.spec)
        if not match:
            return 0.0
        mode, tags_str = match.group(1), match.group(2)
        tags = {t.strip() for t in tags_str.split(",") if t.strip()}
        if not tags:
            return 0.0
        # Query entire map (large radius)
        found = workspace.query_annotations(
            position=(workspace.map_size // 2, workspace.map_size // 2),
            radius=workspace.map_size,
            tags=tags if mode == "any" else None,
        )
        if mode == "all":
            # Filter: only keep annotations that have ALL tags
            found = [a for a in found if tags.issubset(a.tags)]
        if found:
            return min(max(a.intensity for a in found), 1.0)
        return 0.0


@dataclass
class GameForwardModel:
    """Forward model describing expected outcomes of a skill execution.

    Used for planning and verification of skill effects.
    """

    expected_artifacts: frozenset[str] = frozenset()
    invariants: tuple[str, ...] = ()
    success_predicate: str = ""


@dataclass
class GameSkill:
    """A tactical skill with triggers, salience, and action generation.

    Skills are the primary units of tactical behavior. Each skill has one or more
    triggers that determine when it should activate, and a controller that
    generates the actual action.
    """

    id: str
    name: str
    triggers: tuple[GameTrigger, ...]
    salience: float = 1.0
    no_go_traces: int = 0
    prior: float = 0.5
    forward_model: GameForwardModel | None = None
    precision: float = 1.0
    cost_estimate: float = 0.0
    controller: str = ""  # name of the controller function
    provenance: str = ""
