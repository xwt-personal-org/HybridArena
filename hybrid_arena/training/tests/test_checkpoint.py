"""Tests for training checkpoint utilities."""

import torch

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    network = ActorCritic(hidden_dim=16)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        network,
        optimizer,
        PPOConfig(hidden_dim=16, device="cpu"),
        global_step=123,
        metrics={"win_rate": 0.5},
    )

    loaded = load_checkpoint(checkpoint_path)
    assert loaded["global_step"] == 123
    assert loaded["metrics"]["win_rate"] == 0.5

    restored = ActorCritic(hidden_dim=16)
    load_checkpoint(checkpoint_path, network=restored, map_location="cpu")

    for expected, actual in zip(network.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
