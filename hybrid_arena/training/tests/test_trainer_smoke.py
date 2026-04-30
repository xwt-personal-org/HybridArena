"""Smoke tests for the PPO trainer."""

import pytest

from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.training.trainer import Trainer


@pytest.mark.smoke
def test_trainer_smoke_returns_episode_rewards():
    config = PPOConfig(
        total_timesteps=32,
        num_steps=8,
        max_steps=20,
        map_size=16,
        team_size=2,
        device="cpu",
    )
    metrics = Trainer(config, algo_type="ppo").train()

    assert isinstance(metrics, dict)
    assert "episode_rewards" in metrics


@pytest.mark.smoke
def test_trainer_smoke_saves_checkpoint(tmp_path):
    config = PPOConfig(
        total_timesteps=16,
        num_steps=8,
        max_steps=20,
        map_size=16,
        team_size=2,
        device="cpu",
    )
    metrics = Trainer(
        config,
        algo_type="ppo",
        checkpoint_dir=str(tmp_path),
        save_interval=1,
    ).train()

    assert "global_timestep" in metrics
    assert "fps" in metrics
    assert "last_policy_loss" in metrics
    assert "last_value_loss" in metrics
    assert list(tmp_path.glob("ppo_seed42_step*.pt"))


@pytest.mark.smoke
def test_trainer_smoke_num_envs_2():
    config = PPOConfig(
        total_timesteps=16,
        num_envs=2,
        num_steps=4,
        max_steps=20,
        map_size=16,
        team_size=2,
        device="cpu",
    )
    trainer = Trainer(config, algo_type="ppo")
    metrics = trainer.train()

    assert trainer.num_agents == 8
    assert metrics["global_timestep"] == 16
