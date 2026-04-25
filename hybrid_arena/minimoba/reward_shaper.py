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
    farm: float = 0.1
    damage: float = 0.01
    heal: float = 0.01
    win: float = 5.0
    lose: float = -5.0
    time_penalty: float = -0.001


DEFAULT_REWARD_CONFIG = RewardConfig()
