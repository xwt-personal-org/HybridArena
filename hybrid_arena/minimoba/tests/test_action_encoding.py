"""Tests for MiniMOBA joint action encoding."""

import pytest

from hybrid_arena.minimoba.action_encoding import (
    N_ACTIONS,
    decode_action,
    encode_action,
    validate_action_components,
)


def test_encode_action_boundaries():
    assert N_ACTIONS == 324
    assert encode_action(0, 0, 0) == 0
    assert encode_action(8, 3, 8) == 323


def test_decode_action_boundaries():
    assert decode_action(0) == (0, 0, 0)
    assert decode_action(323) == (8, 3, 8)


@pytest.mark.parametrize(
    ("move", "skill", "target"),
    [(-1, 0, 0), (9, 0, 0), (0, -1, 0), (0, 4, 0), (0, 0, -1), (0, 0, 9)],
)
def test_validate_action_components_rejects_out_of_bounds(move, skill, target):
    with pytest.raises(ValueError):
        validate_action_components(move, skill, target)


@pytest.mark.parametrize("index", [-1, 324])
def test_decode_action_rejects_out_of_bounds(index):
    with pytest.raises(ValueError):
        decode_action(index)
