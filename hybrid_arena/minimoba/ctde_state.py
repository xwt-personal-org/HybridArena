"""Centralized-training state helpers for MiniMOBA.

The vector built here is intentionally separate from per-agent observations.
Actors continue to consume only local observations; centralized critics and
offline learners may opt into this full-game summary.
"""

from __future__ import annotations

import numpy as np

from hybrid_arena.minimoba.game_engine import GameState

GLOBAL_STATE_DIM = 96


def build_global_state(game_state: GameState) -> np.ndarray:
    """Return a deterministic fixed-size full-game state vector."""
    values: list[float] = [
        game_state.step_count / max(game_state.max_steps, 1),
        game_state.red_kills / 30.0,
        game_state.blue_kills / 30.0,
        (game_state.red_gold - game_state.blue_gold) / 5000.0,
        game_state.red_towers / 3.0,
        game_state.blue_towers / 3.0,
        game_state.red_tower_damage / 5000.0,
        game_state.blue_tower_damage / 5000.0,
        game_state.red_base_damage / 5000.0,
        game_state.blue_base_damage / 5000.0,
    ]

    for agent_id in game_state.possible_agents:
        hero = game_state.heroes.get(agent_id)
        if hero is None:
            values.extend([0.0] * 9)
            continue
        values.extend(
            [
                1.0 if hero.team == "red" else -1.0,
                hero.x / max(game_state.map_size, 1),
                hero.y / max(game_state.map_size, 1),
                hero.hp_ratio,
                hero.mp_ratio,
                hero.level / 15.0,
                hero.gold / 5000.0,
                hero.damage_dealt / 5000.0,
                float(hero.alive),
            ]
        )

    for structure in sorted(game_state.structures.values(), key=lambda item: item.structure_id):
        values.extend(
            [
                1.0 if structure.team == "red" else -1.0,
                1.0 if structure.structure_type == "base" else 0.5,
                structure.x / max(game_state.map_size, 1),
                structure.y / max(game_state.map_size, 1),
                structure.hp_ratio,
            ]
        )

    state = np.zeros((GLOBAL_STATE_DIM,), dtype=np.float32)
    usable = min(len(values), GLOBAL_STATE_DIM)
    state[:usable] = np.asarray(values[:usable], dtype=np.float32)
    return np.clip(state, -1.0, 1.0)
