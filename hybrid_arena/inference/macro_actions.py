"""Macro action vocabulary for the planner MVP."""

from __future__ import annotations

MACRO_ACTIONS = [
    "DEFEND_OBJECTIVE",
    "GROUP_MID",
    "RETREAT_FARM",
    "INVADE_JUNGLE",
    "PUSH_LANE",
    "BAIT_FIGHT",
    "group_mid",
    "push_nearest_tower",
    "retreat",
    "farm_safe",
    "protect_support",
    "force_teamfight",
    "split_push",
]

ACTION_ALIASES = {
    "group_mid": "GROUP_MID",
    "push_nearest_tower": "PUSH_LANE",
    "retreat": "RETREAT_FARM",
    "farm_safe": "RETREAT_FARM",
    "force_teamfight": "BAIT_FIGHT",
    "split_push": "PUSH_LANE",
    "protect_support": "DEFEND_OBJECTIVE",
}


def validate_macro_action(action: str) -> str:
    action = action.strip()
    return action if action in MACRO_ACTIONS else "group_mid"


def canonical_macro_action(action: str) -> str:
    action = validate_macro_action(action)
    if action in ACTION_ALIASES:
        return ACTION_ALIASES[action]
    return action
