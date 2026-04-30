"""Tests for synchronous multi-environment runner."""

import numpy as np

from hybrid_arena.training.vector_runner import SyncParallelEnvRunner


def test_vector_runner_shapes():
    runner = SyncParallelEnvRunner(
        num_envs=2,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 20},
        seed=42,
    )
    obs, infos = runner.reset()

    assert len(runner.possible_agents) == 8
    assert len(obs) == 8
    assert len(infos) == 8

    actions = {
        agent: np.array([0, 3, 8], dtype=np.int64)
        for agent in runner.possible_agents
    }
    next_obs, rewards, terminations, truncations, next_infos = runner.step(actions)

    assert len(next_obs) == 8
    assert len(rewards) == 8
    assert set(terminations) == set(runner.possible_agents)
    assert set(truncations) == set(runner.possible_agents)
    assert len(next_infos) == 8
