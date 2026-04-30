"""Curriculum Learning scheduler for MiniMOBA.

Design:
    - Adaptive difficulty based on recent win rate (not fixed schedule).
    - Three controllable dimensions: opponent strength, map complexity, reward density.
    - Auto level-up when win_rate > 0.70, level-down when < 0.30.

Reference: interview talking point
    "Curriculum Learning的关键不是'按计划递增难度'，而是'根据当前agent
     能力自适应调节'。我用agent的胜率作为能力指标：胜率>70%就升难度，
     <30%就降难度。"
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class CurriculumLevel:
    """A single difficulty level definition."""

    name: str
    map_size: int
    team_size: int
    skills_enabled: bool
    reward_density: str  # "dense" | "normal" | "sparse"
    opponent_type: str  # "rule_based" | "self_play_weak" | "self_play"
    max_steps: int
    # Optional: scale reward magnitudes
    reward_scale: float = 1.0


DEFAULT_LEVELS = [
    CurriculumLevel(
        name="entry",
        map_size=16,
        team_size=2,
        skills_enabled=False,
        reward_density="dense",
        opponent_type="rule_based",
        max_steps=500,
        reward_scale=1.5,
    ),
    CurriculumLevel(
        name="basic",
        map_size=24,
        team_size=3,
        skills_enabled=True,
        reward_density="dense",
        opponent_type="rule_based",
        max_steps=750,
        reward_scale=1.2,
    ),
    CurriculumLevel(
        name="standard",
        map_size=32,
        team_size=4,
        skills_enabled=True,
        reward_density="normal",
        opponent_type="self_play_weak",
        max_steps=1000,
        reward_scale=1.0,
    ),
    CurriculumLevel(
        name="advanced",
        map_size=32,
        team_size=4,
        skills_enabled=True,
        reward_density="sparse",
        opponent_type="self_play",
        max_steps=1000,
        reward_scale=0.8,
    ),
]


class CurriculumScheduler:
    """Adaptive curriculum scheduler.

    Monitors win rate over a sliding window and adjusts difficulty level.
    """

    def __init__(
        self,
        levels: list[CurriculumLevel] | None = None,
        window_size: int = 20,
        promote_threshold: float = 0.70,
        demote_threshold: float = 0.30,
    ):
        self.levels = levels if levels is not None else copy.deepcopy(DEFAULT_LEVELS)
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold

        self.current_level = 0
        self._win_history: list[float] = []
        self._level_history: list[int] = []

    def update(self, win: bool | float) -> int | None:
        """Record a game result and possibly adjust level.

        Args:
            win: True/False or win_rate scalar in [0, 1].

        Returns:
            New level index if changed, None otherwise.
        """
        self._win_history.append(float(win))
        if len(self._win_history) > self.window_size:
            self._win_history.pop(0)

        if len(self._win_history) < self.window_size:
            return None

        recent_wr = float(np.mean(self._win_history))

        if recent_wr >= self.promote_threshold and self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self._win_history = []  # reset window after transition
            self._level_history.append(self.current_level)
            print(f"[Curriculum] Level UP -> {self.current_level} ({self.levels[self.current_level].name})")
            return self.current_level

        if recent_wr <= self.demote_threshold and self.current_level > 0:
            self.current_level -= 1
            self._win_history = []
            self._level_history.append(self.current_level)
            print(f"[Curriculum] Level DOWN -> {self.current_level} ({self.levels[self.current_level].name})")
            return self.current_level

        return None

    def get_env_config(self) -> dict:
        """Return environment kwargs for the current level."""
        lvl = self.levels[self.current_level]
        return {
            "map_size": lvl.map_size,
            "team_size": lvl.team_size,
            "max_steps": lvl.max_steps,
            "fog_of_war": True,
        }

    def get_reward_config(self) -> dict:
        """Return reward config scaled for current level."""
        lvl = self.levels[self.current_level]
        base = {
            "kill": 1.0,
            "death": -0.8,
            "assist": 0.3,
            "tower": 2.0,
            "tower_lost": -2.0,
            "farm": 0.1,
            "damage": 0.01,
            "heal": 0.01,
            "win": 5.0,
            "lose": -5.0,
            "time_penalty": -0.001,
        }
        if lvl.reward_density == "sparse":
            # Reduce intermediate rewards, emphasize win/lose
            base["kill"] *= 0.3
            base["death"] *= 0.3
            base["farm"] *= 0.2
            base["damage"] *= 0.2
            base["heal"] *= 0.2
        elif lvl.reward_density == "dense":
            # Amplify intermediate feedback for learning
            base["kill"] *= 1.5
            base["farm"] *= 2.0
            base["damage"] *= 1.5

        # Apply global scale
        for k in base:
            base[k] *= lvl.reward_scale

        return base

    @property
    def level_name(self) -> str:
        return self.levels[self.current_level].name

    def state_dict(self) -> dict:
        return {
            "current_level": self.current_level,
            "win_history": list(self._win_history),
            "level_history": list(self._level_history),
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_level = state["current_level"]
        self._win_history = list(state.get("win_history", []))
        self._level_history = list(state.get("level_history", []))
