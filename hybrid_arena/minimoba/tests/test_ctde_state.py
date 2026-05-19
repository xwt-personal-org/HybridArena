import numpy as np
import pytest

from hybrid_arena.algorithms.marl.ctde import CTDEBatch
from hybrid_arena.minimoba.ctde_state import GLOBAL_STATE_DIM, build_global_state
from hybrid_arena.minimoba.env import parallel_env


def test_global_state_is_fixed_and_deterministic():
    env = parallel_env(map_size=16, team_size=2, max_steps=10)
    obs, _ = env.reset(seed=7)
    assert env.game_state is not None
    state_a = build_global_state(env.game_state)
    state_b = build_global_state(env.game_state)
    assert state_a.shape == (GLOBAL_STATE_DIM,)
    assert state_a.dtype == np.float32
    np.testing.assert_allclose(state_a, state_b)
    assert "global_state" not in obs["red_0"]
    assert "ctde_global_state" not in obs["red_0"]


def test_env_exposes_ctde_global_state_outside_actor_obs():
    env = parallel_env(map_size=16, team_size=2, max_steps=10)
    obs, _ = env.reset(seed=8)
    global_state = env.get_global_state()
    assert global_state.shape == (GLOBAL_STATE_DIM,)
    assert "global_state" not in obs["red_0"]


def test_ctde_batch_rejects_global_state_in_actor_observation():
    batch = CTDEBatch(
        actor_observations=[{"global_state": np.zeros(4)}],
        critic_global_states=np.zeros((1, GLOBAL_STATE_DIM), dtype=np.float32),
        actions=np.zeros((1, 3), dtype=np.int64),
        rewards=np.zeros((1,), dtype=np.float32),
        dones=np.zeros((1,), dtype=bool),
    )
    with pytest.raises(ValueError, match="Actor observations"):
        batch.validate_decentralized_actor_inputs()
