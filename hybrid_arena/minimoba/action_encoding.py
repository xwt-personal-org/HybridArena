"""Utilities for MiniMOBA joint action encoding."""

from __future__ import annotations

N_MOVE = 9
N_SKILL = 4
N_TARGET = 9
N_ACTIONS = N_MOVE * N_SKILL * N_TARGET


def validate_action_components(move: int, skill: int, target: int) -> None:
    """Raise ValueError if any action component is outside its valid range."""
    if not 0 <= move < N_MOVE:
        raise ValueError(f"move must be in [0, {N_MOVE - 1}], got {move}")
    if not 0 <= skill < N_SKILL:
        raise ValueError(f"skill must be in [0, {N_SKILL - 1}], got {skill}")
    if not 0 <= target < N_TARGET:
        raise ValueError(f"target must be in [0, {N_TARGET - 1}], got {target}")


def encode_action(move: int, skill: int, target: int) -> int:
    """Return flat joint action index for move, skill, and target."""
    validate_action_components(move, skill, target)
    return move * (N_SKILL * N_TARGET) + skill * N_TARGET + target


def decode_action(index: int) -> tuple[int, int, int]:
    """Return move, skill, and target components from a flat joint action index."""
    if not 0 <= index < N_ACTIONS:
        raise ValueError(f"action index must be in [0, {N_ACTIONS - 1}], got {index}")
    move = index // (N_SKILL * N_TARGET)
    skill = (index % (N_SKILL * N_TARGET)) // N_TARGET
    target = index % N_TARGET
    return move, skill, target
