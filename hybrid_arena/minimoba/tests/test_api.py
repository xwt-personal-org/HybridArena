"""PettingZoo API compliance tests for MiniMOBA."""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from hybrid_arena.minimoba.env import parallel_env


def test_parallel_api():
    """Run PettingZoo's official parallel API compliance suite."""
    env = parallel_env(map_size=16, team_size=2, max_steps=50)
    parallel_api_test(env, num_cycles=200)


def test_seed_determinism():
    """Same seed must produce identical trajectories."""
    env1 = parallel_env(map_size=16, team_size=2, max_steps=30)
    env2 = parallel_env(map_size=16, team_size=2, max_steps=30)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    for agent in env1.possible_agents:
        for key in obs1[agent]:
            np.testing.assert_array_almost_equal(obs1[agent][key], obs2[agent][key], decimal=5)

    # One step
    actions1 = {a: env1.action_space(a).sample() for a in env1.agents}
    actions2 = {a: actions1[a] for a in env2.agents}

    nprng = np.random.RandomState(42)
    for a in env1.agents:
        if a not in actions2:
            actions2[a] = nprng.randint(0, 9, 3).astype(np.int64)
        if a not in actions1:
            actions1[a] = actions2[a]

    obs1, rew1, _, _, _ = env1.step(actions1)
    obs2, rew2, _, _, _ = env2.step(actions2)

    for agent in set(env1.possible_agents) & set(obs1.keys()):
        for key in obs1[agent]:
            np.testing.assert_array_almost_equal(obs1[agent][key], obs2[agent][key], decimal=4)


def test_action_mask_valid():
    """Action mask must leave at least some actions valid."""
    env = parallel_env(map_size=16, team_size=2, max_steps=30)
    obs, _ = env.reset(seed=1)
    for agent in env.agents:
        mask = obs[agent]["action_mask"]
        assert mask.sum() > 0, f"{agent} has no valid actions"


def test_observation_shapes():
    """Verify observation shapes match expected dimensions."""
    env = parallel_env(map_size=16, team_size=2, max_steps=30)
    obs, _ = env.reset(seed=1)
    for agent in env.agents:
        o = obs[agent]
        assert o["local_map"].shape == (11, 11, 11), f"{agent} local_map shape {o['local_map'].shape}"
        assert o["self_state"].shape == (20,), f"{agent} self_state shape {o['self_state'].shape}"
        assert o["teammate_states"].shape == (3, 15), f"{agent} teammate shape {o['teammate_states'].shape}"
        assert o["global_info"].shape == (10,), f"{agent} global shape {o['global_info'].shape}"
        assert o["action_mask"].shape == (324,), f"{agent} mask shape {o['action_mask'].shape}"


def test_rendering():
    """RGB array rendering should not crash."""
    pytest.importorskip("pygame", reason="pygame not installed")
    env = parallel_env(map_size=16, team_size=2, max_steps=10, render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    env.close()
