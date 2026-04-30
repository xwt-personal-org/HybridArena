"""Tests for MiniMOBA joint action masks."""

import numpy as np

from hybrid_arena.minimoba.action_encoding import N_ACTIONS, encode_action
from hybrid_arena.minimoba.env import parallel_env


def test_action_mask_shape_and_dtype():
    env = parallel_env(map_size=16, team_size=2, max_steps=30)
    obs, _ = env.reset(seed=42)
    mask = obs["red_0"]["action_mask"]

    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == np.int8
    assert set(np.unique(mask)).issubset({0, 1})


def test_no_attack_only_allows_target_8():
    env = parallel_env(map_size=16, team_size=2, max_steps=30)
    obs, _ = env.reset(seed=42)
    mask = obs["red_0"]["action_mask"]

    for move in range(9):
        valid_targets = [
            target
            for target in range(9)
            if mask[encode_action(move, 3, target)] == 1
        ]
        assert valid_targets == [8]


def test_stunned_hero_only_noop_action():
    env = parallel_env(map_size=16, team_size=2, max_steps=30)
    env.reset(seed=42)
    hero = env.game_state.heroes["red_0"]
    hero.stunned_turns = 1

    mask = env.game_state.get_observation("red_0")["action_mask"]

    assert int(mask.sum()) == 1
    assert mask[encode_action(0, 3, 8)] == 1


def test_skill_cooldown_masks_skill_actions():
    env = parallel_env(map_size=16, team_size=2, max_steps=30)
    env.reset(seed=42)
    hero = env.game_state.heroes["red_0"]
    hero.skill_1_cd = 2

    mask = env.game_state.get_observation("red_0")["action_mask"]

    for move in range(9):
        for target in range(9):
            assert mask[encode_action(move, 1, target)] == 0
