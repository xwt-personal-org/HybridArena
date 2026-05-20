"""Convert planner macro actions into valid low-level MiniMOBA actions."""

from __future__ import annotations

import numpy as np

from hybrid_arena.inference.macro_actions import canonical_macro_action
from hybrid_arena.minimoba.action_encoding import decode_action, encode_action


class MacroActionAdapter:
    def __init__(self, macro_action: str = "group_mid"):
        self.macro_action = macro_action

    def act(self, obs: dict) -> np.ndarray:
        mask = obs["action_mask"]
        candidates = self._candidate_actions()
        for move, skill, target in candidates:
            flat = encode_action(move, skill, target)
            if mask[flat] == 1:
                return np.array([move, skill, target], dtype=np.int64)

        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return np.array([0, 3, 8], dtype=np.int64)
        return np.array(decode_action(int(valid[0])), dtype=np.int64)

    def to_reward_bias(self) -> dict[str, float]:
        macro = canonical_macro_action(self.macro_action, allow_aliases=True)
        if macro == "DEFEND_OBJECTIVE":
            return {"death": -0.2, "tower_lost": -0.5}
        if macro == "GROUP_MID":
            return {"assist": 0.2, "kill": 0.1}
        if macro == "RETREAT_FARM":
            return {"death": -0.5, "time_penalty": 0.05}
        if macro == "INVADE_JUNGLE":
            return {"damage": 0.1, "kill": 0.2}
        if macro == "PUSH_LANE":
            return {"tower": 0.5, "objective_tower_damage_team": 0.2}
        if macro == "BAIT_FIGHT":
            return {"assist": 0.3, "kill": 0.2}
        return {}

    def to_action_mask_bias(self, action_mask: np.ndarray, *, strength: float = 0.25) -> np.ndarray:
        bias = np.zeros_like(action_mask, dtype=np.float32)
        for move, skill, target in self._candidate_actions():
            flat = encode_action(move, skill, target)
            if action_mask[flat] > 0:
                bias[flat] = strength
        return bias

    def _candidate_actions(self) -> list[tuple[int, int, int]]:
        macro = canonical_macro_action(self.macro_action, allow_aliases=True)
        if self.macro_action == "retreat" or macro == "RETREAT_FARM":
            return [(8, 3, 8), (7, 3, 8), (6, 3, 8), (0, 3, 8)]
        if self.macro_action == "push_nearest_tower" or macro == "PUSH_LANE":
            return [(3, 0, 8), (2, 0, 8), (3, 3, 8), (0, 0, 8)]
        if self.macro_action == "force_teamfight" or macro == "BAIT_FIGHT":
            return [(3, 1, 0), (3, 2, 0), (0, 0, 0), (3, 3, 8)]
        if self.macro_action == "split_push" or macro == "INVADE_JUNGLE":
            return [(2, 0, 8), (4, 0, 8), (2, 3, 8), (4, 3, 8)]
        if self.macro_action == "protect_support" or macro == "DEFEND_OBJECTIVE":
            return [(0, 3, 8), (7, 3, 8), (8, 3, 8)]
        return [(0, 3, 8), (3, 3, 8), (5, 3, 8)]
