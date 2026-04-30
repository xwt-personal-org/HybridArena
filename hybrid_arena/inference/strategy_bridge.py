"""StrategyToRLBridge: Convert LLM high-level strategy into DRL reward signals.

Two complementary mechanisms:
    1. Reward shaping: adjust reward weights based on strategy type.
    2. Goal position injection: add navigation bonus toward target positions.
"""

from __future__ import annotations

import numpy as np

# Strategy -> reward weight modifiers
STRATEGY_REWARD_MODIFIERS = {
    "团战": {
        "kill": 2.0,
        "death": 1.5,
        "assist": 2.0,
        "tower": 1.0,
        "farm": 0.3,
        "grouping_bonus": 0.5,
    },
    "分推": {
        "kill": 0.5,
        "death": 0.5,
        "assist": 0.3,
        "tower": 3.0,
        "farm": 1.5,
        "spread_bonus": 0.3,
    },
    "发育": {
        "kill": 0.3,
        "death": 2.0,
        "assist": 0.2,
        "tower": 0.3,
        "farm": 3.0,
        "safe_farming_bonus": 0.5,
    },
    "防守": {
        "kill": 0.5,
        "death": 2.5,
        "assist": 1.0,
        "tower": 0.5,
        "tower_lost": 3.0,
        "near_tower_bonus": 0.5,
    },
    "抓人": {
        "kill": 3.0,
        "death": 1.0,
        "assist": 1.5,
        "tower": 0.5,
        "farm": 0.5,
        "isolation_bonus": 0.5,
    },
}

DEFAULT_MODIFIERS = STRATEGY_REWARD_MODIFIERS["团战"]


class StrategyToRLBridge:
    """Bridge LLM strategy to DRL reward shaping and navigation goals."""

    def __init__(self, base_reward_config: dict | None = None):
        self.base_reward_config = base_reward_config or {}

    def get_reward_modifiers(self, strategy: str) -> dict:
        """Return reward weight modifiers for a given strategy."""
        return STRATEGY_REWARD_MODIFIERS.get(strategy, DEFAULT_MODIFIERS).copy()

    def apply_reward_shaping(
        self,
        strategy: str,
        base_rewards: dict[str, float],
    ) -> dict[str, float]:
        """Apply strategy-specific reward shaping to a dict of base rewards.

        Args:
            strategy: Strategy name (团战/分推/发育/防守/抓人).
            base_rewards: Dict of reward components (kill, death, etc.).

        Returns:
            Modified rewards.
        """
        modifiers = self.get_reward_modifiers(strategy)
        shaped = {}
        for key, val in base_rewards.items():
            shaped[key] = val * modifiers.get(key, 1.0)
        return shaped

    def get_goal_positions(
        self,
        assignments: dict,
        hero_configs: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract target positions from LLM assignments.

        Args:
            assignments: Dict with "target_positions" key containing role->[x,y].
            hero_configs: Optional mapping of hero_id -> role.

        Returns:
            Dict mapping hero_id or role to target position array.
        """
        goals = {}
        targets = assignments.get("target_positions", {})
        for role, pos in targets.items():
            goals[role] = np.array(pos, dtype=np.float32)
        return goals

    def navigation_bonus(
        self,
        current_pos: np.ndarray,
        goal_pos: np.ndarray,
        alpha: float = 0.01,
    ) -> float:
        """Compute negative distance reward toward goal.

        Args:
            current_pos: (2,) current position.
            goal_pos: (2,) target position.
            alpha: Scaling factor.

        Returns:
            Reward bonus (negative distance * alpha).
        """
        dist = np.linalg.norm(current_pos - goal_pos)
        return -alpha * dist
