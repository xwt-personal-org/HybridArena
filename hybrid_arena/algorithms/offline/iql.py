"""Discrete IQL smoke-test objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteIQLTrainer:
    """Small implicit Q-learning loss helper for CPU tests."""

    def __init__(self, input_dim: int, n_actions: int = 324, hidden_dim: int = 64, expectile: float = 0.7):
        self.q_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_actions))
        self.v_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.expectile = expectile

    def expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weights = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return (weights * diff.pow(2)).mean()

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.q_net(observations)
        chosen_q = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        values = self.v_net(observations).squeeze(1)
        value_loss = self.expectile_loss(chosen_q.detach() - values)
        q_loss = F.mse_loss(chosen_q, target_q)
        return q_loss + value_loss
