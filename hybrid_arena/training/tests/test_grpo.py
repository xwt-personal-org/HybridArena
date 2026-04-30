"""Tests for GRPO trainer."""

from __future__ import annotations

import torch

from hybrid_arena.training.grpo_trainer import GRPOTrainer


class TestGRPOTrainer:
    def test_mock_mode_when_no_transformers(self):
        trainer = GRPOTrainer(device="cpu")
        # If transformers is not installed, should fall back to mock
        assert trainer.tokenizer is not None

    def test_generate_group_mock(self):
        trainer = GRPOTrainer(num_generations=4, device="cpu")
        prompts = trainer._build_prompt("Test game state")
        responses = trainer.generate_group(prompts)
        assert len(responses) == 4

    def test_compute_rewards(self):
        trainer = GRPOTrainer(device="cpu")
        responses = ['{"strategy":"团战"}', '{"bad":true}', "not json"]
        rewards = trainer.compute_rewards(responses, trainer.default_reward_fn)
        assert rewards.shape == (3,)
        assert rewards[0].item() == 1.0
        assert rewards[2].item() == -1.0

    def test_grpo_loss_shape(self):
        trainer = GRPOTrainer(num_generations=4, device="cpu")
        prompts = ["state1", "state2"]

        def mock_reward_fn(response: str) -> float:
            return 0.5

        loss, info = trainer.compute_grpo_loss(prompts, mock_reward_fn)
        assert loss.dim() == 0
        assert "avg_reward" in info

    def test_group_relative_advantage(self):
        rewards = torch.tensor([1.0, 0.5, 0.0, -0.5])
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r
        assert abs(advantages.sum().item()) < 1e-4  # should sum to ~0
