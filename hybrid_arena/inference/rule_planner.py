"""Rule-based macro planner baseline."""

from __future__ import annotations

from hybrid_arena.inference.macro_actions import validate_macro_action


class RulePlanner:
    def plan(self, state) -> str:
        if state.ally_summary.get("low_hp", 0) >= max(1, state.ally_summary.get("alive", 0) // 2):
            return "retreat"
        if state.objective_summary.get("enemy_tower_hp", 0.0) < 1200.0:
            return "push_nearest_tower"
        if state.score_summary.get("gold_diff", 0.0) > 300 and state.enemy_summary.get("visible", 0) > 0:
            return "force_teamfight"
        return validate_macro_action("group_mid")
