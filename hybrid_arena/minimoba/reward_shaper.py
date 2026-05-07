"""Configurable reward shaper for MiniMOBA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    kill: float = 1.0
    death: float = -0.8
    assist: float = 0.3
    tower: float = 2.0
    tower_lost: float = -2.0
    base: float = 3.0
    farm: float = 0.1
    damage: float = 0.01
    heal: float = 0.01
    win: float = 5.0
    lose: float = -5.0
    time_penalty: float = -0.001

    # Objective shaping (Phase F13)
    objective_enabled: bool = False
    objective_tower_damage_team: float = 0.001
    objective_base_damage_team: float = 0.003
    objective_base_exposed_team: float = 1.0
    objective_step_cap_team: float = 0.25


DEFAULT_REWARD_CONFIG = RewardConfig()
