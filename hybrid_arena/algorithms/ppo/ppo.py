"""Vanilla PPO implementation (CleanRL style)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.algorithms.ppo.config import PPOConfig


class PPO:
    """Vanilla Proximal Policy Optimization (Schulman et al., 2017).

    Implements:
    - Standard clipped surrogate objective
    - GAE advantage estimation (computed externally in buffer)
    - Adaptive entropy coefficient (linear/cosine decay)
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.network = ActorCritic(hidden_dim=config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate, eps=1e-5
        )

        self.global_step = 0
        self.total_timesteps = config.total_timesteps

    def get_entropy_coef(self, global_step: int, total_steps: int) -> float:
        """Compute entropy coefficient based on schedule."""
        c = self.config
        if c.entropy_schedule == "linear_decay":
            progress = min(global_step / max(total_steps, 1), 1.0)
            return c.entropy_start + (c.entropy_end - c.entropy_start) * progress
        elif c.entropy_schedule == "cosine_decay":
            import math

            progress = min(global_step / max(total_steps, 1), 1.0)
            return c.entropy_end + 0.5 * (c.entropy_start - c.entropy_end) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            return c.entropy_coef

    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_masks: torch.Tensor | None,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute PPO loss for a minibatch.

        Args:
            obs: Dict of batched observations.
            actions: (B, 3) actions taken.
            old_log_probs: (B,) log probs from rollout policy.
            advantages: (B,) GAE advantages (normalized).
            returns: (B,) GAE returns (value targets).
            action_masks: (B, 324) or None.
            old_values: (B,) value estimates saved during rollout.

        Returns:
            total_loss, info dict.
        """
        _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
            obs, actions, action_masks
        )

        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
        )
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values,
            -self.config.clip_eps,
            self.config.clip_eps,
        )
        value_losses = (new_values - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        # Entropy bonus
        entropy_coef = self.get_entropy_coef(self.global_step, self.total_timesteps)
        entropy_loss = -entropy.mean()

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + entropy_coef * entropy_loss
        )

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()

        info = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "entropy_coef": entropy_coef,
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
        }
        return total_loss, info

    def update(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        old_values: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one PPO update epoch.

        Args:
            obs: Dict of batched observations.
            actions: (B_q, 3).
            old_log_probs: (B_q,).
            advantages: (B_q,).
            returns: (B_q,).
            action_masks: (B_q, 324) or None.
            old_values: (B_q,) rollout value estimates.

        Returns:
            Dict of training metrics.
        """
        total_batch = actions.shape[0]
        if old_values is None:
            raise ValueError("old_values is required for PPO clipped value loss")
        indices = torch.randperm(total_batch)

        epoch_info = {}
        n_updates = 0

        for start in range(0, total_batch, self.config.minibatch_size):
            end = start + self.config.minibatch_size
            mb_idx = indices[start:end]

            loss, info = self.compute_loss(
                {k: v[mb_idx] for k, v in obs.items()},
                actions[mb_idx],
                old_log_probs[mb_idx],
                advantages[mb_idx],
                returns[mb_idx],
                action_masks[mb_idx] if action_masks is not None else None,
                old_values[mb_idx],
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            for k, v in info.items():
                epoch_info[k] = epoch_info.get(k, 0.0) + v
            n_updates += 1

        for k in epoch_info:
            epoch_info[k] /= n_updates

        self.global_step += 1
        return epoch_info

    def get_action(
        self,
        obs: dict[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return log prob + value.

        Returns:
            action, log_prob, value (all detached for rollout collection).
        """
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(
                obs, action_mask=action_mask
            )
        return action, log_prob, value
