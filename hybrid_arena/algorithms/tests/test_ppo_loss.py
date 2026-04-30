"""Tests for PPO and DualClipPPO loss contracts."""

import torch

from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.algorithms.ppo.ppo import PPO
from hybrid_arena.algorithms.ppo.ppo_dualclip import DualClipPPO
from hybrid_arena.minimoba.action_encoding import N_ACTIONS


class _FixedValueNetwork:
    def get_action_and_value(self, obs, actions, action_masks):
        batch_size = actions.shape[0]
        new_log_probs = torch.zeros(batch_size, dtype=torch.float32)
        entropy = torch.zeros(batch_size, dtype=torch.float32)
        new_values = torch.ones(batch_size, dtype=torch.float32, requires_grad=True)
        return actions, new_log_probs, entropy, new_values


def _make_obs(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "local_map": torch.zeros(batch_size, 11, 11, 11, dtype=torch.float32),
        "self_state": torch.zeros(batch_size, 20, dtype=torch.float32),
        "teammate_states": torch.zeros(batch_size, 3, 15, dtype=torch.float32),
        "global_info": torch.zeros(batch_size, 10, dtype=torch.float32),
    }


def test_ppo_update_accepts_old_values():
    config = PPOConfig(
        hidden_dim=16,
        minibatch_size=4,
        total_timesteps=16,
        device="cpu",
    )
    ppo = PPO(config)
    batch_size = 4
    obs = _make_obs(batch_size)
    actions = torch.zeros(batch_size, 3, dtype=torch.int64)
    old_log_probs = torch.zeros(batch_size, dtype=torch.float32)
    advantages = torch.ones(batch_size, dtype=torch.float32)
    returns = torch.ones(batch_size, dtype=torch.float32)
    action_masks = torch.ones(batch_size, N_ACTIONS, dtype=torch.int8)
    old_values = torch.zeros(batch_size, dtype=torch.float32)

    info = ppo.update(
        obs,
        actions,
        old_log_probs,
        advantages,
        returns,
        action_masks,
        old_values,
    )

    assert "value_loss" in info


def test_ppo_value_loss_uses_old_values_for_clipping():
    config = PPOConfig(hidden_dim=16, clip_eps=0.2, device="cpu")
    ppo = PPO(config)
    ppo.network = _FixedValueNetwork()
    batch_size = 2

    _, info = ppo.compute_loss(
        obs=_make_obs(batch_size),
        actions=torch.zeros(batch_size, 3, dtype=torch.int64),
        old_log_probs=torch.zeros(batch_size, dtype=torch.float32),
        advantages=torch.ones(batch_size, dtype=torch.float32),
        returns=torch.full((batch_size,), 4.0, dtype=torch.float32),
        action_masks=torch.ones(batch_size, N_ACTIONS, dtype=torch.int8),
        old_values=torch.zeros(batch_size, dtype=torch.float32),
    )

    assert info["value_loss"] == torch.tensor(7.22).item()


def test_dual_clip_fraction_is_finite():
    config = PPOConfig(
        hidden_dim=16,
        minibatch_size=4,
        total_timesteps=16,
        device="cpu",
    )
    ppo = DualClipPPO(config)
    batch_size = 4

    info = ppo.update(
        _make_obs(batch_size),
        torch.zeros(batch_size, 3, dtype=torch.int64),
        torch.zeros(batch_size, dtype=torch.float32),
        torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float32),
        torch.ones(batch_size, dtype=torch.float32),
        torch.ones(batch_size, N_ACTIONS, dtype=torch.int8),
        torch.zeros(batch_size, dtype=torch.float32),
    )

    assert torch.isfinite(torch.tensor(info["dual_clip_fraction"]))
    assert 0.0 <= info["dual_clip_fraction"] <= 1.0
