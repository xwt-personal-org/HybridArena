"""Tests for MAPPO."""

from __future__ import annotations

import torch

from hybrid_arena.algorithms.mappo.mappo import MAPPO, CentralizedCritic
from hybrid_arena.algorithms.ppo.config import PPOConfig


class TestCentralizedCritic:
    def test_output_shape(self):
        critic = CentralizedCritic(n_agents=8, self_dim=20, global_dim=10)
        self_states = torch.randn(4, 8, 20)
        global_infos = torch.randn(4, 8, 10)
        values = critic(self_states, global_infos)
        assert values.shape == (4,)

    def test_single_batch(self):
        critic = CentralizedCritic(n_agents=2, self_dim=5, global_dim=3)
        self_states = torch.randn(2, 5)
        global_infos = torch.randn(2, 3)
        values = critic(self_states, global_infos)
        assert values.shape == (1,)


class TestMAPPO:
    def test_get_action_shapes(self):
        config = PPOConfig(hidden_dim=16, device="cpu")
        mappo = MAPPO(config, n_agents=8)

        obs = {
            "local_map": torch.randn(8, 11, 11, 11),
            "self_state": torch.randn(8, 20),
            "teammate_states": torch.randn(8, 3, 15),
            "global_info": torch.randn(8, 10),
            "action_mask": torch.ones(8, 324, dtype=torch.int8),
        }
        action, log_prob, value = mappo.get_action(obs)
        assert action.shape == (8, 3)
        assert log_prob.shape == (8,)
        assert value.shape == (8,)

    def test_compute_loss_shapes(self):
        config = PPOConfig(hidden_dim=16, device="cpu", minibatch_size=64)
        mappo = MAPPO(config, n_agents=8)

        batch_size = 128  # T=16, N=8
        obs = {
            "local_map": torch.randn(batch_size, 11, 11, 11),
            "self_state": torch.randn(batch_size, 20),
            "teammate_states": torch.randn(batch_size, 3, 15),
            "global_info": torch.randn(batch_size, 10),
            "action_mask": torch.ones(batch_size, 324, dtype=torch.int8),
        }
        actions = torch.stack([
            torch.randint(0, 9, (batch_size,)),
            torch.randint(0, 4, (batch_size,)),
            torch.randint(0, 9, (batch_size,)),
        ], dim=1)
        old_logp = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        old_values = torch.randn(batch_size)

        loss, info = mappo.compute_loss(
            obs,
            actions,
            old_logp,
            advantages,
            returns,
            obs["action_mask"],
            old_values,
        )
        assert loss.dim() == 0
        assert "policy_loss" in info
        assert "value_loss" in info

    def test_update_runs(self):
        config = PPOConfig(
            hidden_dim=16, device="cpu", minibatch_size=32, n_epochs=2
        )
        mappo = MAPPO(config, n_agents=4)

        batch_size = 32  # T=8, N=4
        obs = {
            "local_map": torch.randn(batch_size, 11, 11, 11),
            "self_state": torch.randn(batch_size, 20),
            "teammate_states": torch.randn(batch_size, 3, 15),
            "global_info": torch.randn(batch_size, 10),
            "action_mask": torch.ones(batch_size, 324, dtype=torch.int8),
        }
        actions = torch.stack([
            torch.randint(0, 9, (batch_size,)),
            torch.randint(0, 4, (batch_size,)),
            torch.randint(0, 9, (batch_size,)),
        ], dim=1)
        old_logp = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        old_values = torch.randn(batch_size)

        info = mappo.update(
            obs,
            actions,
            old_logp,
            advantages,
            returns,
            obs["action_mask"],
            old_values,
        )
        assert "policy_loss" in info
        assert "approx_kl" in info
