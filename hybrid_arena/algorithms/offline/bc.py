"""Tiny behavior cloning trainer for offline replay smoke tests."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_arena.minimoba.action_encoding import N_ACTIONS, encode_action


def flatten_observation(obs: dict) -> np.ndarray:
    parts = [
        np.asarray(obs["local_map"], dtype=np.float32).reshape(-1),
        np.asarray(obs["self_state"], dtype=np.float32).reshape(-1),
        np.asarray(obs["teammate_states"], dtype=np.float32).reshape(-1),
        np.asarray(obs["global_info"], dtype=np.float32).reshape(-1),
    ]
    return np.concatenate(parts).astype(np.float32)


class _BCPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BehaviorCloningTrainer:
    """Minimal discrete behavior cloning interface."""

    def __init__(self, input_dim: int, lr: float = 3e-3, seed: int = 0):
        torch.manual_seed(seed)
        self.policy = _BCPolicy(input_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    @staticmethod
    def action_to_index(action: list[int] | np.ndarray) -> int:
        move, skill, target = [int(x) for x in action]
        return encode_action(move, skill, target)

    def loss(self, observations: np.ndarray, actions: np.ndarray) -> torch.Tensor:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        return F.cross_entropy(self.policy(obs_tensor), action_tensor)

    def train_epoch(self, observations: np.ndarray, actions: np.ndarray) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(observations, actions)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())
