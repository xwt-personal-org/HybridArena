"""Macro action vocabulary for the planner MVP."""

from __future__ import annotations

MACRO_ACTIONS = [
    "group_mid",
    "push_nearest_tower",
    "retreat",
    "farm_safe",
    "protect_support",
    "force_teamfight",
    "split_push",
]


def validate_macro_action(action: str) -> str:
    action = action.strip()
    return action if action in MACRO_ACTIONS else "group_mid"
