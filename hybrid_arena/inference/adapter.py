"""Convert planner macro actions into valid low-level MiniMOBA actions."""

from __future__ import annotations

import numpy as np

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

    def _candidate_actions(self) -> list[tuple[int, int, int]]:
        if self.macro_action == "retreat":
            return [(8, 3, 8), (7, 3, 8), (6, 3, 8), (0, 3, 8)]
        if self.macro_action == "push_nearest_tower":
            return [(3, 0, 8), (2, 0, 8), (3, 3, 8), (0, 0, 8)]
        if self.macro_action == "force_teamfight":
            return [(3, 1, 0), (3, 2, 0), (0, 0, 0), (3, 3, 8)]
        if self.macro_action == "split_push":
            return [(2, 0, 8), (4, 0, 8), (2, 3, 8), (4, 3, 8)]
        if self.macro_action == "protect_support":
            return [(0, 3, 8), (7, 3, 8), (8, 3, 8)]
        return [(0, 3, 8), (3, 3, 8), (5, 3, 8)]
