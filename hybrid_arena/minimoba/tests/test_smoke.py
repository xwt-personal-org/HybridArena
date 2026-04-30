"""Fast end-to-end smoke tests for the MiniMOBA environment."""

import numpy as np
import pytest

from hybrid_arena.minimoba.env import parallel_env


def _sample_legal_action(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    valid_indices = np.flatnonzero(mask)
    assert valid_indices.size > 0
    action_index = int(rng.choice(valid_indices))
    move = action_index // 36
    skill = (action_index % 36) // 9
    target = action_index % 9
    return np.array([move, skill, target], dtype=np.int64)


@pytest.mark.smoke
def test_env_smoke_2v2_20_steps():
    rng = np.random.default_rng(42)
    env = parallel_env(map_size=16, team_size=2, max_steps=20)
    observations, _ = env.reset(seed=42)

    for _ in range(20):
        if not env.agents:
            break
        actions = {
            agent: _sample_legal_action(observations[agent]["action_mask"], rng)
            for agent in env.agents
        }
        observations, _, terminations, truncations, _ = env.step(actions)
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()
