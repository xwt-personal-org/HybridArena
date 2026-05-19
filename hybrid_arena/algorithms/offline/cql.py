"""Discrete CQL smoke-test objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteCQLTrainer:
    """Small conservative Q-learning loss helper for CPU tests."""

    def __init__(self, input_dim: int, n_actions: int = 324, hidden_dim: int = 64, alpha: float = 0.5):
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.alpha = alpha

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values = self.q_net(observations)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask <= 0, -1e8)
        chosen_q = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        bellman = F.mse_loss(chosen_q, target_q)
        conservative = torch.logsumexp(q_values, dim=1).mean() - chosen_q.mean()
        return bellman + self.alpha * conservative
