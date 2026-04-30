"""QMIX: Monotonic value function factorisation for cooperative MARL.

Reference:
    Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep
    Multi-Agent Reinforcement Learning", ICML 2018.

Key concept (interview talking point):
    "QMIX enforces ∂Q_tot/∂Q_i ≥ 0 via a mixing network with positive weights.
    This guarantees that increasing any agent's local Q improves the joint Q,
    but it cannot express situations where one agent's Q must decrease for
    the team to win — that's why QPLEX/WQMIX were proposed."
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_arena.algorithms.networks import MapEncoder, StateEncoder


class AgentQNetwork(nn.Module):
    """Independent Q-network for one agent.

    Outputs Q-values for all 324 action combinations (move×skill×target).
    """

    def __init__(self, hidden_dim: int = 48, n_actions: int = 324):
        super().__init__()
        self.map_encoder = MapEncoder(hidden_dim=hidden_dim)
        self.state_encoder = StateEncoder(hidden_dim=hidden_dim)
        feature_dim = hidden_dim + hidden_dim * 3
        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return Q-values for all actions.

        Returns:
            q_values: (batch, n_actions)
        """
        map_feat = self.map_encoder(obs["local_map"])
        state_feat = self.state_encoder(
            obs["self_state"], obs["teammate_states"], obs["global_info"]
        )
        features = torch.cat([map_feat, state_feat], dim=-1)
        return self.q_head(features)


class MixingNetwork(nn.Module):
    """Mixing network with monotonicity constraint (positive weights).

    Q_tot = f(Q_1, Q_2, ..., Q_n; s) where f is monotonic in each Q_i.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int = 10,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Hypernetworks generate positive weights
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        q_values: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q_values: (batch, n_agents)
            global_state: (batch, state_dim)

        Returns:
            q_tot: (batch,)
        """
        batch_size = q_values.shape[0]

        # Layer 1: (batch, n_agents) -> (batch, hidden_dim)
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.hidden_dim)

        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)

        # Layer 2: (batch, hidden_dim) -> (batch, 1)
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class QMIXAgent:
    """QMIX training agent (off-policy with experience replay).

    Simplified for MiniMOBA: parameter-sharing Q-networks,
    epsilon-greedy exploration, target network soft update.
    """

    def __init__(
        self,
        n_agents: int = 8,
        hidden_dim: int = 48,
        gamma: float = 0.99,
        lr: float = 5e-4,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.q_network = AgentQNetwork(hidden_dim=hidden_dim).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.eval()

        self.mixer = MixingNetwork(n_agents=n_agents, hidden_dim=32).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mixer.eval()

        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_actions(
        self,
        obs: dict[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
        eval_mode: bool = False,
    ) -> torch.Tensor:
        """Epsilon-greedy action selection.

        Args:
            obs: Batched obs dict (batch, ...).
            action_mask: (batch, 324) binary mask.
            eval_mode: If True, always greedy.

        Returns:
            actions: (batch, 3) MultiDiscrete actions.
        """
        batch_size = obs["self_state"].shape[0]
        if not eval_mode and torch.rand(1).item() < self.epsilon:
            # Random actions respecting mask
            actions = torch.zeros(batch_size, 3, dtype=torch.int64, device=self.device)
            if action_mask is not None:
                for i in range(batch_size):
                    valid = action_mask[i].nonzero(as_tuple=True)[0]
                    if len(valid) > 0:
                        idx = valid[torch.randint(0, len(valid), (1,)).item()].item()
                        actions[i, 0] = idx // 36
                        actions[i, 1] = (idx % 36) // 9
                        actions[i, 2] = idx % 9
            else:
                actions[:, 0] = torch.randint(0, 9, (batch_size,), device=self.device)
                actions[:, 1] = torch.randint(0, 4, (batch_size,), device=self.device)
                actions[:, 2] = torch.randint(0, 9, (batch_size,), device=self.device)
            return actions

        with torch.no_grad():
            q_values = self.q_network(obs)  # (batch, 324)
            if action_mask is not None:
                q_values = q_values.masked_fill(~action_mask.bool(), -1e8)
            actions_flat = q_values.argmax(dim=-1)  # (batch,)

        # Decode flat action index to (move, skill, target)
        move = actions_flat // 36
        skill = (actions_flat % 36) // 9
        target = actions_flat % 9
        return torch.stack([move, skill, target], dim=1)

    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        dones: torch.Tensor,
        action_masks: torch.Tensor | None,
        next_action_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute QMIX TD loss.

        Args:
            obs: Batched obs (B, ...) — already flattened over (T, N_agents).
            actions: (B, 3) actions taken.
            rewards: (B,) rewards.
            next_obs: Batched next obs (B, ...).
            dones: (B,) done flags.
            action_masks: (B, 324) masks for obs.
            next_action_masks: (B, 324) masks for next_obs.

        Returns:
            loss, info dict.
        """
        batch_size = actions.shape[0]

        # Encode actions to flat index
        actions_flat = actions[:, 0] * 36 + actions[:, 1] * 9 + actions[:, 2]

        # Current Q-values
        q_values = self.q_network(obs)  # (B, 324)
        q_taken = q_values.gather(1, actions_flat.unsqueeze(1)).squeeze(1)  # (B,)

        # Reshape to (T, N_agents) for mixer — need to know T
        # For simplicity, assume B is always divisible by n_agents
        if batch_size % self.n_agents != 0:
            raise ValueError(f"Batch size {batch_size} not divisible by n_agents {self.n_agents}")
        num_steps = batch_size // self.n_agents

        q_taken = q_taken.view(num_steps, self.n_agents)
        global_state = obs["global_info"].view(num_steps, self.n_agents, -1)[:, 0, :]  # (T, 10)
        q_tot = self.mixer(q_taken, global_state)  # (T,)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_q_network(next_obs)  # (B, 324)
            if next_action_masks is not None:
                next_q = next_q.masked_fill(~next_action_masks.bool(), -1e8)
            next_q_max = next_q.max(dim=-1)[0]  # (B,)
            next_q_max = next_q_max.view(num_steps, self.n_agents)
            next_global = next_obs["global_info"].view(num_steps, self.n_agents, -1)[:, 0, :]
            q_tot_target = self.target_mixer(next_q_max, next_global)  # (T,)

        rewards_sum = rewards.view(num_steps, self.n_agents).sum(dim=1)  # (T,)
        dones_any = dones.view(num_steps, self.n_agents).max(dim=1)[0]  # (T,)
        target = rewards_sum + self.gamma * (1.0 - dones_any) * q_tot_target

        loss = F.mse_loss(q_tot, target)

        info = {
            "loss": loss.item(),
            "q_tot_mean": q_tot.mean().item(),
            "target_mean": target.mean().item(),
        }
        return loss, info

    def update(self, batch: dict) -> dict:
        """Run one QMIX update step."""
        loss, info = self.compute_loss(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["dones"],
            batch.get("action_masks"),
            batch.get("next_action_masks"),
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            10.0,
        )
        self.optimizer.step()

        # Soft update target networks
        self._soft_update(self.q_network, self.target_q_network)
        self._soft_update(self.mixer, self.target_mixer)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        info["epsilon"] = self.epsilon

        return info

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for sp, tp in zip(source.parameters(), target.parameters(), strict=True):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
