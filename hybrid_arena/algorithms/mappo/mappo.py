"""MAPPO: Multi-Agent PPO with CTDE centralized critic.

Reference:
    Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
    Games", 2022.

Design:
    - Parameter-sharing actor (all agents share actor weights).
    - Centralized critic: input = concatenated self_states + global_infos of
      all agents.  This is the "CT" in CTDE.
    - Decentralized execution: each agent uses only local obs at inference.

Interview talking point:
    "CTDE means centralized training with decentralized execution.
    During training the critic sees the global state (all agents' observations),
    but at test time each actor only uses its own local view."
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.algorithms.ppo.ppo import PPO


class CentralizedCritic(nn.Module):
    """Critic that observes concatenated states of all agents.

    Input: all agents' self_state (20-d) + global_info (10-d) concatenated.
    Output: a single joint value V(s) expanded to all agents.
    """

    def __init__(
        self,
        n_agents: int,
        self_dim: int = 20,
        global_dim: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = n_agents * (self_dim + global_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.n_agents = n_agents

    def forward(
        self,
        self_states: torch.Tensor,
        global_infos: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            self_states: (batch, n_agents, self_dim) or (n_agents, self_dim).
            global_infos: (batch, n_agents, global_dim) or (n_agents, global_dim).

        Returns:
            values: (batch,) or scalar.
        """
        if self_states.dim() == 2:
            self_states = self_states.unsqueeze(0)
            global_infos = global_infos.unsqueeze(0)

        x = torch.cat([self_states, global_infos], dim=-1)  # (B, N, self_dim+global_dim)
        x = x.view(x.shape[0], -1)  # (B, N*(self_dim+global_dim))
        return self.net(x).squeeze(-1)  # (B,)


class MAPPO(PPO):
    """Multi-Agent PPO with centralized critic.

    Args:
        config: PPO hyperparameters.
        n_agents: Number of agents (8 for 4v4).
    """

    def __init__(self, config: PPOConfig, n_agents: int = 8):
        super().__init__(config)
        self.n_agents = n_agents
        # Re-create optimizer to include critic params
        self.critic = CentralizedCritic(
            n_agents=n_agents,
            hidden_dim=config.hidden_dim * 2,
        ).to(self.device)
        params = list(self.network.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate, eps=1e-5)

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
        """Compute MAPPO loss: shared actor + centralized critic."""
        batch_size = actions.shape[0]
        # Actor forward (same as PPO, per-agent)
        _, new_log_probs, entropy, _ = self.network.get_action_and_value(
            obs, actions, action_masks
        )

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
        )
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Centralized critic forward
        # Recover (T, N_agents, ...) from flattened batch
        num_steps = batch_size // self.n_agents
        self_states = obs["self_state"].view(num_steps, self.n_agents, -1)
        global_infos = obs["global_info"].view(num_steps, self.n_agents, -1)

        centralized_values = self.critic(self_states, global_infos)  # (T,)
        # Expand to per-agent values: (T, N_agents)
        centralized_values = centralized_values.unsqueeze(1).expand(-1, self.n_agents)
        centralized_values = centralized_values.reshape(-1)  # (T*N_agents,)

        value_pred_clipped = old_values + torch.clamp(
            centralized_values - old_values,
            -self.config.clip_eps,
            self.config.clip_eps,
        )
        value_losses = (centralized_values - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

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

    def get_action(
        self,
        obs: dict[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (actor) and compute centralized value (critic).

        Returns:
            action, log_prob, value.
        """
        with torch.no_grad():
            action, log_prob, _, _ = self.network.get_action_and_value(
                obs, action_mask=action_mask
            )
            # Centralized critic
            n = obs["self_state"].shape[0]  # should be n_agents
            self_states = obs["self_state"].view(1, n, -1)
            global_infos = obs["global_info"].view(1, n, -1)
            value = self.critic(self_states, global_infos)  # (1,)
            value = value.expand(n)  # (n_agents,)
        return action, log_prob, value
