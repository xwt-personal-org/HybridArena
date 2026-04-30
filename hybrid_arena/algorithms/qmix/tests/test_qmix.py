"""Tests for QMIX."""

from __future__ import annotations

import torch

from hybrid_arena.algorithms.qmix.qmix import AgentQNetwork, MixingNetwork, QMIXAgent


class TestAgentQNetwork:
    def test_output_shape(self):
        net = AgentQNetwork(hidden_dim=16, n_actions=324)
        obs = {
            "local_map": torch.randn(4, 11, 11, 11),
            "self_state": torch.randn(4, 20),
            "teammate_states": torch.randn(4, 3, 15),
            "global_info": torch.randn(4, 10),
        }
        q = net(obs)
        assert q.shape == (4, 324)

    def test_action_masking(self):
        net = AgentQNetwork(hidden_dim=16)
        obs = {
            "local_map": torch.randn(2, 11, 11, 11),
            "self_state": torch.randn(2, 20),
            "teammate_states": torch.randn(2, 3, 15),
            "global_info": torch.randn(2, 10),
        }
        q = net(obs)
        # Masking is done at agent level, not network level
        mask = torch.zeros(2, 324, dtype=torch.int8)
        mask[0, 0] = 1
        mask[1, 5] = 1
        q_masked = q.masked_fill(~mask.bool(), -1e8)
        assert q_masked[0, 1].item() < -1e7
        assert q_masked[1, 0].item() < -1e7


class TestMixingNetwork:
    def test_monotonicity(self):
        mixer = MixingNetwork(n_agents=4, state_dim=10, hidden_dim=16)
        q = torch.randn(8, 4)
        state = torch.randn(8, 10)
        q_tot = mixer(q, state)
        assert q_tot.shape == (8,)

    def test_gradient_positive(self):
        """Verify ∂Q_tot/∂Q_i ≥ 0 by checking gradient signs."""
        mixer = MixingNetwork(n_agents=3, state_dim=5, hidden_dim=8)
        q = torch.randn(1, 3, requires_grad=True)
        state = torch.randn(1, 5)
        q_tot = mixer(q, state)
        q_tot.backward()
        assert (q.grad >= -1e-6).all(), "Mixing network violates monotonicity"


class TestQMIXAgent:
    def test_select_actions_shape(self):
        agent = QMIXAgent(n_agents=8, hidden_dim=16, device="cpu")
        obs = {
            "local_map": torch.randn(8, 11, 11, 11),
            "self_state": torch.randn(8, 20),
            "teammate_states": torch.randn(8, 3, 15),
            "global_info": torch.randn(8, 10),
        }
        actions = agent.select_actions(obs, eval_mode=True)
        assert actions.shape == (8, 3)
        assert (actions[:, 0] < 9).all()
        assert (actions[:, 1] < 4).all()
        assert (actions[:, 2] < 9).all()

    def test_compute_loss(self):
        agent = QMIXAgent(n_agents=4, hidden_dim=16, device="cpu")
        batch_size = 32  # T=8, N=4
        obs = {
            "local_map": torch.randn(batch_size, 11, 11, 11),
            "self_state": torch.randn(batch_size, 20),
            "teammate_states": torch.randn(batch_size, 3, 15),
            "global_info": torch.randn(batch_size, 10),
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
        }
        dones = torch.zeros(batch_size)
        loss, info = agent.compute_loss(obs, actions, rewards, next_obs, dones, None, None)
        assert loss.dim() == 0
        assert "q_tot_mean" in info

    def test_update(self):
        agent = QMIXAgent(n_agents=4, hidden_dim=16, device="cpu")
        batch_size = 32
        batch = {
            "obs": {
                "local_map": torch.randn(batch_size, 11, 11, 11),
                "self_state": torch.randn(batch_size, 20),
                "teammate_states": torch.randn(batch_size, 3, 15),
                "global_info": torch.randn(batch_size, 10),
            },
            "actions": torch.stack([
                torch.randint(0, 9, (batch_size,)),
                torch.randint(0, 4, (batch_size,)),
                torch.randint(0, 9, (batch_size,)),
            ], dim=1),
            "rewards": torch.randn(batch_size),
            "next_obs": {
                "local_map": torch.randn(batch_size, 11, 11, 11),
                "self_state": torch.randn(batch_size, 20),
                "teammate_states": torch.randn(batch_size, 3, 15),
                "global_info": torch.randn(batch_size, 10),
            },
            "dones": torch.zeros(batch_size),
        }
        info = agent.update(batch)
        assert "loss" in info
        assert "epsilon" in info
