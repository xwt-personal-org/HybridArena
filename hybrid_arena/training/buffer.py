"""On-policy rollout buffer with GAE computation.

Design: stores (T, N_agents, ...) tensors from a single environment with
N_agents agents acting simultaneously. Flattens to (T*N_agents, ...) for PPO.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_agents: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        self.observations: list[dict[str, np.ndarray]] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

        self.step_idx = 0

    def reset(self):
        self.step_idx = 0
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(
        self,
        obs_batch: dict[str, np.ndarray],
        action_batch: np.ndarray,
        log_prob_batch: np.ndarray,
        reward_batch: np.ndarray,
        done_batch: np.ndarray,
        value_batch: np.ndarray,
    ):
        """Store one step of rollout.

        All arrays have shape (N_agents, ...).
        """
        self.observations.append(obs_batch)
        self.actions.append(action_batch)
        self.log_probs.append(log_prob_batch)
        self.rewards.append(reward_batch)
        self.dones.append(done_batch)
        self.values.append(value_batch)
        self.step_idx += 1

    @property
    def full(self) -> bool:
        return self.step_idx >= self.num_steps

    def get_batch(
        self,
        next_values: np.ndarray,
        next_dones: np.ndarray,
    ) -> dict:
        """Compute GAE and return flat batch.

        Args:
            next_values: (N_agents,) values at step T+1.
            next_dones: (N_agents,) done flags at step T+1.

        Returns:
            Dict with obs, actions, log_probs, advantages, returns.
        """
        n_agents = self.num_agents
        n_steps = len(self.rewards)  # actual steps collected

        # Stack: (N_steps, N_agents, ...)
        rew = np.stack(self.rewards, axis=0)   # (N_steps, N_agents)
        don = np.stack(self.dones, axis=0)     # (N_steps, N_agents)
        val = np.stack(self.values, axis=0)    # (N_steps, N_agents)
        act = np.stack(self.actions, axis=0)    # (N_steps, N_agents, 3)
        logp = np.stack(self.log_probs, axis=0) # (N_steps, N_agents)

        # GAE per agent
        advantages = np.zeros((n_steps, n_agents), dtype=np.float32)
        for a in range(n_agents):
            lastgaelam = 0.0
            for t in reversed(range(n_steps)):
                if t < n_steps - 1:
                    next_non_terminal = 1.0 - don[t + 1, a]
                    nv = val[t + 1, a]
                else:
                    next_non_terminal = 1.0 - next_dones[a]
                    nv = next_values[a]
                delta = rew[t, a] + self.gamma * nv * next_non_terminal - val[t, a]
                lastgaelam = delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
                advantages[t, a] = lastgaelam

        returns = advantages + val  # (T, N_agents)

        # Flatten to (T * N_agents, ...)
        batch_obs = self._flatten_obs()
        return {
            "obs": batch_obs,
            "actions": torch.tensor(act.reshape(-1, act.shape[-1]), dtype=torch.int64, device=self.device),
            "log_probs": torch.tensor(logp.reshape(-1), dtype=torch.float32, device=self.device),
            "advantages": torch.tensor(advantages.reshape(-1), dtype=torch.float32, device=self.device),
            "returns": torch.tensor(returns.reshape(-1), dtype=torch.float32, device=self.device),
        }

    def _flatten_obs(self) -> dict[str, torch.Tensor]:
        """Flatten observation dict from T steps to (T * N_agents, ...)."""
        keys = ["local_map", "self_state", "teammate_states", "global_info"]
        flattened: dict[str, list[np.ndarray]] = defaultdict(list)

        n_steps = len(self.observations)
        for t in range(n_steps):
            obs_t = self.observations[t]  # {key: (N_agents, ...)}
            for a in range(self.num_agents):
                for key in keys:
                    if key in obs_t:
                        flattened[key].append(obs_t[key][a])

        return {
            "local_map": torch.tensor(np.stack(flattened["local_map"]), dtype=torch.float32, device=self.device),
            "self_state": torch.tensor(np.stack(flattened["self_state"]), dtype=torch.float32, device=self.device),
            "teammate_states": torch.tensor(np.stack(flattened["teammate_states"]), dtype=torch.float32, device=self.device),
            "global_info": torch.tensor(np.stack(flattened["global_info"]), dtype=torch.float32, device=self.device),
        }
