"""WEN-86 verification: MiniMOBA 4v4 PettingZoo env, action space, and action mask."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo.test import parallel_api_test

from hybrid_arena.minimoba.action_encoding import (
    N_ACTIONS,
    N_MOVE,
    N_SKILL,
    N_TARGET,
    decode_action,
    encode_action,
)
from hybrid_arena.minimoba.env import parallel_env
from hybrid_arena.minimoba.hero import DEFAULT_HERO_ASSIGNMENTS

DEFAULT_4V4_KWARGS = {"map_size": 32, "team_size": 4, "max_steps": 50}


def _sample_legal_action(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    valid_indices = np.flatnonzero(mask)
    assert valid_indices.size > 0
    action_index = int(rng.choice(valid_indices))
    move, skill, target = decode_action(action_index)
    return np.array([move, skill, target], dtype=np.int64)


def test_4v4_agent_configuration():
    """Red/blue each field 4 heroes; 8 parallel agents total."""
    env = parallel_env(**DEFAULT_4V4_KWARGS)
    assert env.possible_agents == [
        "red_0",
        "red_1",
        "red_2",
        "red_3",
        "blue_0",
        "blue_1",
        "blue_2",
        "blue_3",
    ]
    obs, _ = env.reset(seed=42)
    assert len(env.agents) == 8
    assert set(obs.keys()) == set(env.possible_agents)


def test_4v4_default_hero_assignments():
    """Default lineup: tank + 2× dps + support per team."""
    env = parallel_env(**DEFAULT_4V4_KWARGS)
    env.reset(seed=42)
    for team in ("red", "blue"):
        roles = [env.game_state.heroes[f"{team}_{i}"].config_name for i in range(4)]
        assert roles == DEFAULT_HERO_ASSIGNMENTS


def test_4v4_action_space_multidiscrete():
    """Each agent exposes MultiDiscrete([9, 4, 9]) = 324 joint actions."""
    env = parallel_env(**DEFAULT_4V4_KWARGS)
    for agent in env.possible_agents:
        action_space = env.action_space(agent)
        assert isinstance(action_space, spaces.MultiDiscrete)
        assert action_space.nvec.tolist() == [N_MOVE, N_SKILL, N_TARGET]
        assert int(np.prod(action_space.nvec)) == N_ACTIONS == 324


def test_4v4_action_mask_shape_and_validity():
    """324-d mask is binary and leaves at least one legal action per agent."""
    env = parallel_env(**DEFAULT_4V4_KWARGS)
    obs, _ = env.reset(seed=1)
    for agent in env.agents:
        mask = obs[agent]["action_mask"]
        assert mask.shape == (N_ACTIONS,)
        assert set(np.unique(mask)).issubset({0, 1})
        assert mask.sum() > 0, f"{agent} has no valid actions"


def test_4v4_action_mask_aligns_with_encoding():
    """Every masked-in action decodes to in-range MultiDiscrete components."""
    env = parallel_env(**DEFAULT_4V4_KWARGS)
    obs, _ = env.reset(seed=7)
    for agent in env.agents:
        mask = obs[agent]["action_mask"]
        for index in np.flatnonzero(mask):
            move, skill, target = decode_action(int(index))
            assert mask[encode_action(move, skill, target)] == 1
            if skill == 3:
                assert target == 8


def test_4v4_observation_shapes():
    """Observation dict keys and shapes match 4v4 spec (3 teammate slots)."""
    env = parallel_env(**DEFAULT_4V4_KWARGS)
    obs, _ = env.reset(seed=1)
    for agent in env.agents:
        o = obs[agent]
        assert o["local_map"].shape == (11, 11, 11)
        assert o["self_state"].shape == (20,)
        assert o["teammate_states"].shape == (3, 15)
        assert o["global_info"].shape == (10,)
        assert o["action_mask"].shape == (324,)


def test_4v4_parallel_api():
    """PettingZoo official parallel API compliance on full 4v4 config."""
    env = parallel_env(map_size=16, team_size=4, max_steps=30)
    parallel_api_test(env, num_cycles=100)


@pytest.mark.smoke
def test_4v4_smoke_reset_step_observe():
    """End-to-end smoke: reset → masked legal steps → observe/reward/terminal keys."""
    rng = np.random.default_rng(42)
    env = parallel_env(map_size=16, team_size=4, max_steps=20)
    observations, _ = env.reset(seed=42)

    saw_rewards = False
    for _ in range(20):
        if not env.agents:
            break
        actions = {
            agent: _sample_legal_action(observations[agent]["action_mask"], rng)
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)
        assert all(agent in rewards for agent in env.possible_agents)
        saw_rewards = saw_rewards or any(rewards.values())
        if any(terminations.values()) or any(truncations.values()):
            break

    assert saw_rewards or env.is_game_over
    env.close()


def test_4v4_timeout_terminates_episode():
    """Episode ends via termination (not truncation) when max_steps is reached."""
    env = parallel_env(map_size=16, team_size=4, max_steps=5)
    env.reset(seed=42)
    terminations: dict[str, bool] = {}
    for _ in range(6):
        actions = {a: np.array([0, 3, 8], dtype=np.int64) for a in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
        if not env.agents:
            break

    assert env.game_state.terminal_reason == "timeout"
    assert all(terminations[a] for a in env.game_state.possible_agents)
    assert not any(truncations.values())
