"""Prompt templates for macro planner providers."""

from __future__ import annotations

import json


def build_macro_prompt(planner_state) -> str:
    state_payload = planner_state.to_dict() if hasattr(planner_state, "to_dict") else planner_state
    return (
        "Return JSON only with fields macro_action, reasoning, reward_bias, action_mask_bias. "
        "macro_action must be one of DEFEND_OBJECTIVE, GROUP_MID, RETREAT_FARM, "
        "INVADE_JUNGLE, PUSH_LANE, BAIT_FIGHT. "
        "reward_bias and action_mask_bias must be numeric dictionaries. "
        f"PlannerState={json.dumps(state_payload, sort_keys=True, default=str)}"
    )
