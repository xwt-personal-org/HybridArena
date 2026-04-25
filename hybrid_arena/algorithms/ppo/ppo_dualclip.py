"""Dual-clip PPO (Honor of Kings / AAAI 2020).

Key Insight:
    Standard PPO only clips the upper bound of the ratio. In large-batch distributed
    training, the ratio can also become very small due to policy lag. Dual-clip adds
    a lower bound 'c' on the negative-advantage region, preventing vanishing gradients
    from stale samples.

Reference:
    Ye et al., "Mastering Complex Control in MOBA Games with Deep Reinforcement
    Learning", AAAI 2020.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hybrid_arena.algorithms.ppo.ppo import PPO


class DualClipPPO(PPO):
    """PPO with dual-clip lower bound for negative advantages.

    When advantage < 0:
        L = max(min(ratio * A, clip(ratio) * A), c * A)

    This prevents the ratio from dropping too low when the current policy has
    moved significantly away from the rollout policy.
    """

    def __init__(self, config):
        super().__init__(config)
        self.dual_clip_c = config.dual_clip_c

    def compute_loss(
        self,
        obs,
        actions,
        old_log_probs,
        advantages,
        returns,
        action_masks=None,
    ):
        _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
            obs, actions, action_masks
        )

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
        )
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages

        # Standard clip — upper bound
        policy_loss = torch.min(surrogate1, surrogate2)

        # Dual clip — lower bound for negative advantages
        dual_clip_mask = (advantages < 0).float()
        dual_clip_value = self.dual_clip_c * advantages
        policy_loss = (
            dual_clip_mask * torch.max(policy_loss, dual_clip_value)
            + (1.0 - dual_clip_mask) * policy_loss
        )
        policy_loss = -policy_loss.mean()

        # Value loss (clipped)
        value_pred_clipped = new_values + torch.clamp(
            new_values - new_values.detach(),
            -self.config.clip_eps,
            self.config.clip_eps,
        )
        value_loss = 0.5 * torch.max(
            F.mse_loss(new_values, returns),
            F.mse_loss(value_pred_clipped, returns),
        )

        entropy_coef = self.get_entropy_coef(self.global_step, self.total_timesteps)
        entropy_loss = -entropy.mean()

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + entropy_coef * entropy_loss
        )

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean().item()
            clip_frac = (
                (ratio - 1.0).abs() > self.config.clip_eps
            ).float().mean().item()
            dual_frac = (
                dual_clip_mask * (policy_loss.detach() == dual_clip_value).float()
            ).mean().item()

        info = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "entropy_coef": entropy_coef,
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
            "dual_clip_fraction": dual_frac,
        }
        return total_loss, info
