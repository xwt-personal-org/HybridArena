"""Rule-based agent — finite state machine baseline for MiniMOBA.

States: patrol → engage → retreat → patrol
"""

from __future__ import annotations

import numpy as np


class RuleBasedAgent:
    """Finite state machine AI that serves as baseline for RL algorithms.

    Heuristics:
    - Patrol: move toward center of map, auto-attack nearest enemy in range.
    - Engage: when enemies detected and HP > 50%, chase and use skills aggressively.
    - Retreat: when HP < 20%, move toward own base and avoid fights.
    """

    def __init__(self):
        self.state = "patrol"
        self._target_pos: tuple[int, int] | None = None

    def act(self, obs: dict) -> np.ndarray:
        hp_ratio = obs["self_state"][0]
        mask = obs.get("action_mask")
        if mask is None:
            mask = np.ones(324, dtype=np.int8)

        # Detect nearby enemies
        has_enemy = self._has_visible_enemy(obs)

        # State transition
        if hp_ratio < 0.2:
            self.state = "retreat"
        elif has_enemy and hp_ratio > 0.5:
            self.state = "engage"
        elif has_enemy:
            self.state = "retreat"
        else:
            self.state = "patrol"

        if self.state == "patrol":
            return self._patrol_action(obs, mask)
        elif self.state == "engage":
            return self._engage_action(obs, mask)
        else:
            return self._retreat_action(obs, mask)

    def _has_visible_enemy(self, obs: dict) -> bool:
        local_map = obs.get("local_map")  # (11, 11, 11)
        if local_map is None:
            return False
        # Channel 8 = visible enemies
        return np.any(local_map[:, :, 8] > 0.5)

    def _get_enemy_direction(self, obs: dict) -> int:
        local_map = obs.get("local_map")
        if local_map is None:
            return 0
        enemy_channel = local_map[:, :, 8]
        indices = np.argwhere(enemy_channel > 0.5)
        if len(indices) == 0:
            return 0
        center = np.array([5.0, 5.0])  # center of 11x11
        enemy_pos = indices.mean(axis=0)
        delta = enemy_pos - center
        if np.linalg.norm(delta) < 1.0:
            return 0
        # Convert to 8 directions
        directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        best_dir = 0
        best_dot = -999
        for i, (dx, dy) in enumerate(directions):
            dot = dx * delta[1] + (-dy) * delta[0]
            if dot > best_dot:
                best_dot = dot
                best_dir = i + 1
        return best_dir

    def _patrol_action(self, obs: dict, mask: np.ndarray) -> np.ndarray:
        # Move toward center of map
        move_dir = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 0])
        # Auto-attack nearest enemy if any
        skill = 0
        target = 0
        return np.array([move_dir, skill, target], dtype=np.int64)

    def _engage_action(self, obs: dict, mask: np.ndarray) -> np.ndarray:
        move_dir = self._get_enemy_direction(obs)
        # Use skills if available
        self_state = obs["self_state"]
        skill = 0  # auto-attack
        if self_state[3] < 0.01 and self_state[1] > 0.15:  # skill 1 ready, enough MP
            skill = 1
        elif self_state[4] < 0.01 and self_state[1] > 0.25:  # skill 2 ready
            skill = 2
        target = 0
        return np.array([move_dir, skill, target], dtype=np.int64)

    def _retreat_action(self, obs: dict, mask: np.ndarray) -> np.ndarray:
        # Move away from enemies — opposite of detected enemy direction
        enemy_dir = self._get_enemy_direction(obs)
        if enemy_dir == 0:
            return self._patrol_action(obs, mask)
        # Opposite direction (4 offset in 8-dir circle)
        retreat_dir = ((enemy_dir + 3) % 8) + 1
        return np.array([retreat_dir, 3, 8], dtype=np.int64)
