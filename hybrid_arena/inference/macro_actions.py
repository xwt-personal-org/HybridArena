"""Macro action vocabulary for the planner MVP."""

from __future__ import annotations

import re

CANONICAL_MACRO_ACTIONS = (
    "DEFEND_OBJECTIVE",
    "GROUP_MID",
    "RETREAT_FARM",
    "INVADE_JUNGLE",
    "PUSH_LANE",
    "BAIT_FIGHT",
)

MACRO_ACTIONS = list(CANONICAL_MACRO_ACTIONS)

ACTION_ALIASES = {
    "group_mid": "GROUP_MID",
    "push_nearest_tower": "PUSH_LANE",
    "retreat": "RETREAT_FARM",
    "farm_safe": "RETREAT_FARM",
    "force_teamfight": "BAIT_FIGHT",
    "split_push": "PUSH_LANE",
    "protect_support": "DEFEND_OBJECTIVE",
}

_MACRO_ACTION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _clean_macro_action(action: str) -> str:
    if not isinstance(action, str):
        raise ValueError(f"macro action must be a string, got {type(action).__name__}")
    cleaned = action.strip()
    if not cleaned:
        raise ValueError("macro action must not be empty")
    if _MACRO_ACTION_RE.fullmatch(cleaned) is None:
        raise ValueError(f"malformed macro action: {cleaned!r}")
    return cleaned


def validate_macro_action(action: str, *, allow_aliases: bool = False) -> str:
    cleaned = _clean_macro_action(action)
    if cleaned in CANONICAL_MACRO_ACTIONS:
        return cleaned
    if allow_aliases and cleaned in ACTION_ALIASES:
        return ACTION_ALIASES[cleaned]
    raise ValueError(f"unknown macro action: {cleaned!r}")


def canonical_macro_action(action: str, *, allow_aliases: bool = True) -> str:
    return validate_macro_action(action, allow_aliases=allow_aliases)
