"""Deterministic tactical skills and helper functions."""

from __future__ import annotations

from hybrid_arena.minimoba.game_engine import MOVEMENT_DELTA
from hybrid_arena.minimoba.tactical_runtime.dispatcher import register_controller
from hybrid_arena.minimoba.tactical_runtime.schema import (
    GameForwardModel,
    GameSkill,
    GameTrigger,
)
from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace


def direction_toward(
    target_pos: tuple[int, int],
    current_pos: tuple[int, int],
) -> int:
    """Return the move index (0-8) that moves from current_pos toward target_pos.

    Uses the MOVEMENT_DELTA mapping from game_engine.py.
    Index 0 = stay in place.

    Args:
        target_pos: Target position (x, y).
        current_pos: Current position (x, y).

    Returns:
        Move index in [0, 8].
    """
    tx, ty = target_pos
    cx, cy = current_pos
    dx = tx - cx
    dy = ty - cy

    # Normalize to -1, 0, 1
    if dx > 0:
        ndx = 1
    elif dx < 0:
        ndx = -1
    else:
        ndx = 0
    if dy > 0:
        ndy = 1
    elif dy < 0:
        ndy = -1
    else:
        ndy = 0

    # Find matching move index
    for idx, (mx, my) in MOVEMENT_DELTA.items():
        if mx == ndx and my == ndy:
            return idx
    return 0  # stay in place


def nearest_tagged_region(
    workspace: BattlefieldWorkspace,
    current_pos: tuple[int, int],
    tag: str,
) -> tuple[int, int] | None:
    """Find the position of the nearest annotation with the given tag.

    Args:
        workspace: The battlefield workspace to search.
        current_pos: Current position (x, y).
        tag: Tag to search for.

    Returns:
        Position (x, y) of nearest matching annotation, or None if not found.
    """
    annotations = workspace.query_annotations(
        position=current_pos,
        radius=workspace.map_size,
        tags={tag},
    )
    if not annotations:
        return None

    cx, cy = current_pos
    best_pos = None
    best_dist = float("inf")
    for ann in annotations:
        ax, ay = ann.position
        dist = max(abs(ax - cx), abs(ay - cy))  # Chebyshev distance
        if dist < best_dist:
            best_dist = dist
            best_pos = (ax, ay)
    return best_pos


# --- Controller functions ---


def _controller_retreat(skill, event, workspace, game_state, agent_id):
    """Controller for retreat_when_low: move toward own base (0, 0)."""
    hero = getattr(game_state, "heroes", {}).get(agent_id) if game_state else None
    if hero is None:
        return {"move": 0, "skill": 0, "target": 0}
    move_idx = direction_toward((0, 0), (hero.x, hero.y))
    return {"move": move_idx, "skill": 0, "target": 0}


def _controller_farm(skill, event, workspace, game_state, agent_id):
    """Controller for farm_resources: move toward nearest resource annotation."""
    hero = getattr(game_state, "heroes", {}).get(agent_id) if game_state else None
    if hero is None:
        return {"move": 0, "skill": 0, "target": 0}
    target = nearest_tagged_region(workspace, (hero.x, hero.y), "resource_soon")
    if target is None:
        target = nearest_tagged_region(workspace, (hero.x, hero.y), "objective")
    if target is None:
        return {"move": 0, "skill": 0, "target": 0}
    move_idx = direction_toward(target, (hero.x, hero.y))
    return {"move": move_idx, "skill": 0, "target": 0}


def _controller_vision(skill, event, workspace, game_state, agent_id):
    """Controller for control_vision: move toward nearest danger/vision area."""
    hero = getattr(game_state, "heroes", {}).get(agent_id) if game_state else None
    if hero is None:
        return {"move": 0, "skill": 0, "target": 0}
    target = nearest_tagged_region(workspace, (hero.x, hero.y), "vision_loss")
    if target is None:
        target = nearest_tagged_region(workspace, (hero.x, hero.y), "dangerous")
    if target is None:
        return {"move": 0, "skill": 0, "target": 0}
    move_idx = direction_toward(target, (hero.x, hero.y))
    return {"move": move_idx, "skill": 0, "target": 0}


def _controller_objective(skill, event, workspace, game_state, agent_id):
    """Controller for push_objective: move toward nearest objective."""
    hero = getattr(game_state, "heroes", {}).get(agent_id) if game_state else None
    if hero is None:
        return {"move": 0, "skill": 0, "target": 0}
    target = nearest_tagged_region(workspace, (hero.x, hero.y), "objective")
    if target is None:
        return {"move": 0, "skill": 0, "target": 0}
    move_idx = direction_toward(target, (hero.x, hero.y))
    return {"move": move_idx, "skill": 0, "target": 0}


# Register all controllers
register_controller("retreat", _controller_retreat)
register_controller("farm", _controller_farm)
register_controller("vision", _controller_vision)
register_controller("objective", _controller_objective)


# --- Skill factory ---


def create_tactical_skills(workspace: BattlefieldWorkspace) -> list[GameSkill]:
    """Create the default set of deterministic tactical skills.

    Args:
        workspace: The battlefield workspace (unused directly, but available
                   for future extensions that need workspace-aware skill configs).

    Returns:
        List of 4 tactical skills.
    """
    return [
        GameSkill(
            id="retreat_when_low",
            name="Retreat When Low HP",
            triggers=(
                GameTrigger(kind="health_threshold", spec="below:0.25"),
            ),
            salience=1.0,
            prior=0.5,
            forward_model=GameForwardModel(
                expected_artifacts=frozenset({"position_change"}),
                invariants=("hp > 0",),
                success_predicate="distance_to_base decreased",
            ),
            controller="retreat",
            provenance="deterministic_v1",
        ),
        GameSkill(
            id="farm_resources",
            name="Farm Resources",
            triggers=(
                GameTrigger(kind="annotation_query", spec="any:resource_soon,objective"),
            ),
            salience=0.7,
            prior=0.5,
            forward_model=GameForwardModel(
                expected_artifacts=frozenset({"gold_change"}),
            ),
            controller="farm",
            provenance="deterministic_v1",
        ),
        GameSkill(
            id="control_vision",
            name="Control Vision",
            triggers=(
                GameTrigger(kind="annotation_query", spec="any:vision_loss,dangerous"),
            ),
            salience=0.6,
            prior=0.5,
            controller="vision",
            provenance="deterministic_v1",
        ),
        GameSkill(
            id="push_objective",
            name="Push Objective",
            triggers=(
                GameTrigger(kind="team_state", spec="advantage"),
                GameTrigger(kind="annotation_query", spec="any:objective"),
            ),
            salience=0.8,
            prior=0.5,
            forward_model=GameForwardModel(
                expected_artifacts=frozenset({"tower_damage"}),
            ),
            controller="objective",
            provenance="deterministic_v1",
        ),
    ]
