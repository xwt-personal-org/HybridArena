"""Tests for COMA."""

from __future__ import annotations

import torch

from hybrid_arena.algorithms.coma.coma import COMA, COMACritic


class TestCOMACritic:
    def test_forward_shape(self):
        critic = COMACritic(n_agents=4, n_actions=324, hidden_dim=64)
        ss = torch.randn(8, 4, 20)
        gi = torch.randn(8, 4, 10)
        actions = torch.randint(0, 324, (8, 4))
        q = critic(ss, gi, actions)
        assert q.shape == (8, 4)


class TestCOMA:
    def test_get_action_shapes(self):
        coma = COMA(n_agents=8, hidden_dim=16, device="cpu")
        obs = {
            "local_map": torch.randn(8, 11, 11, 11),
            "self_state": torch.randn(8, 20),
            "teammate_states": torch.randn(8, 3, 15),
            "global_info": torch.randn(8, 10),
            "action_mask": torch.ones(8, 324, dtype=torch.int8),
        }
        action, log_prob, _ = coma.get_action(obs)
        assert action.shape == (8, 3)
        assert log_prob.shape == (8,)

    def test_update_runs(self):
        coma = COMA(n_agents=4, hidden_dim=16, device="cpu")
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
        rewards = torch.randn(batch_size)
        next_obs = {
            "local_map": torch.randn(batch_size, 11, 11, 11),
            "self_state": torch.randn(batch_size, 20),
            "teammate_states": torch.randn(batch_size, 3, 15),
            "global_info": torch.randn(batch_size, 10),
            "action_mask": torch.ones(batch_size, 324, dtype=torch.int8),
        }
        dones = torch.zeros(batch_size)

        info = coma.update(obs, actions, rewards, next_obs, dones, obs["action_mask"])
        assert "critic_loss" in info
        assert "policy_loss" in info
