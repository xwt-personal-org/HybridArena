"""Tests for PPO rollout buffer batching."""

import numpy as np
import torch

from hybrid_arena.training.buffer import RolloutBuffer


def _obs_batch(num_agents: int) -> dict[str, np.ndarray]:
    return {
        "local_map": np.zeros((num_agents, 11, 11, 11), dtype=np.float32),
        "self_state": np.zeros((num_agents, 20), dtype=np.float32),
        "teammate_states": np.zeros((num_agents, 3, 15), dtype=np.float32),
        "global_info": np.zeros((num_agents, 10), dtype=np.float32),
    }


def test_buffer_returns_action_masks_and_old_values():
    buffer = RolloutBuffer(num_steps=2, num_agents=2, device="cpu")

    for step in range(2):
        buffer.add(
            obs_batch=_obs_batch(num_agents=2),
            action_batch=np.full((2, 3), step, dtype=np.int64),
            log_prob_batch=np.full((2,), -0.5, dtype=np.float32),
            reward_batch=np.ones((2,), dtype=np.float32),
            done_batch=np.zeros((2,), dtype=np.float32),
            value_batch=np.array([step, step + 0.25], dtype=np.float32),
            action_mask_batch=np.ones((2, 324), dtype=np.int8),
        )

    batch = buffer.get_batch(
        next_values=np.zeros((2,), dtype=np.float32),
        next_dones=np.ones((2,), dtype=np.float32),
    )

    assert batch["action_masks"].shape == (4, 324)
    assert batch["action_masks"].dtype == torch.int8
    assert batch["old_values"].shape == (4,)
    assert batch["old_values"].dtype == torch.float32
    torch.testing.assert_close(
        batch["old_values"],
        torch.tensor([0.0, 0.25, 1.0, 1.25], dtype=torch.float32),
    )
