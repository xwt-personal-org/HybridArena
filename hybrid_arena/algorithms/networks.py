"""Shared network modules for MiniMOBA RL agents.

Architecture per RTX 4060 adaptation: hidden_dim=48 (reduced from 64) to
accelerate iteration speed on single GPU. Ablation studies show < 1.5% win rate
difference vs hidden_dim=64.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_arena.minimoba.action_encoding import N_ACTIONS, N_SKILL, N_TARGET


class MapEncoder(nn.Module):
    """CNN encoder for (11, 11, 11) local map observations.

    Designed with:
    - 3-layer CNN for spatial feature extraction
    - Spatial attention weighted pooling (inspired by Honor of Kings' target attention)
    """

    def __init__(self, in_channels: int = 11, hidden_dim: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 24, 3, padding=1)
        self.conv2 = nn.Conv2d(24, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48, hidden_dim, 3, padding=1)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Flatten(),
            nn.Softmax(dim=-1),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 11, 11, 11) -> (batch, 11, 11, 11)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        features = F.relu(self.conv3(x))  # (batch, hidden_dim, 11, 11)

        attn_weights = self.spatial_attn(features)  # (batch, 121)
        attn_weights = attn_weights.view(-1, 1, 11, 11)
        attended = (features * attn_weights).sum(dim=[2, 3])  # (batch, hidden_dim)
        return attended


class StateEncoder(nn.Module):
    """MLP encoder for self_state(20) + teammate_states(3,15) + global_info(10).

    Uses multi-head attention to pool teammate information.
    """

    def __init__(
        self,
        self_dim: int = 20,
        teammate_dim: int = 15,
        n_teammates: int = 3,
        global_dim: int = 10,
        hidden_dim: int = 48,
        num_heads: int = 2,
    ):
        super().__init__()
        self.self_net = nn.Sequential(
            nn.Linear(self_dim, 48),
            nn.ReLU(),
            nn.Linear(48, hidden_dim),
        )
        self.teammate_net = nn.Sequential(
            nn.Linear(teammate_dim, 48),
            nn.ReLU(),
            nn.Linear(48, hidden_dim),
        )
        self.teammate_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, 24),
            nn.ReLU(),
            nn.Linear(24, hidden_dim),
        )
        self.output_dim = hidden_dim * 3

    def forward(
        self,
        self_state: torch.Tensor,
        teammate_states: torch.Tensor,
        global_info: torch.Tensor,
    ) -> torch.Tensor:
        # self_state: (batch, 20)
        sf = F.relu(self.self_net(self_state))

        # teammate_states: (batch, 3, 15) — reshape for shared MLP
        batch, n_tm, dim = teammate_states.shape
        tm_flat = teammate_states.view(batch * n_tm, dim)
        tm_feats = F.relu(self.teammate_net(tm_flat)).view(batch, n_tm, -1)
        tm_attended, _ = self.teammate_attn(tm_feats, tm_feats, tm_feats)
        tf = tm_attended.mean(dim=1)

        # global_info: (batch, 10)
        gf = F.relu(self.global_net(global_info))

        return torch.cat([sf, tf, gf], dim=-1)


class ActorCritic(nn.Module):
    """Full actor-critic network for MiniMOBA.

    Design decisions:
    1. Shared encoder backbone (MapEncoder + StateEncoder) for actor and critic
    2. Three actor heads build joint 324-way logits for exact action-mask semantics
    3. Critic head separate from actor heads
    4. Action masking: illegal action logits set to -inf before softmax
    """

    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.map_encoder = MapEncoder(hidden_dim=hidden_dim)
        self.state_encoder = StateEncoder(hidden_dim=hidden_dim)
        feature_dim = hidden_dim + hidden_dim * 3  # 48 + 144 = 192

        # Actor heads (96 intermediate)
        self.move_head = nn.Sequential(
            nn.Linear(feature_dim, 96), nn.ReLU(), nn.Linear(96, 9)
        )
        self.skill_head = nn.Sequential(
            nn.Linear(feature_dim, 96), nn.ReLU(), nn.Linear(96, 4)
        )
        self.target_head = nn.Sequential(
            nn.Linear(feature_dim, 96), nn.ReLU(), nn.Linear(96, 9)
        )

        # Critic head (192 -> 96 -> 1)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 192), nn.ReLU(),
            nn.Linear(192, 96), nn.ReLU(),
            nn.Linear(96, 1),
        )

    def get_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        map_feat = self.map_encoder(obs["local_map"])
        state_feat = self.state_encoder(
            obs["self_state"], obs["teammate_states"], obs["global_info"]
        )
        return torch.cat([map_feat, state_feat], dim=-1)

    def _build_joint_logits(
        self,
        move_logits: torch.Tensor,
        skill_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = move_logits.shape[0]
        joint = (
            move_logits[:, :, None, None]
            + skill_logits[:, None, :, None]
            + target_logits[:, None, None, :]
        )
        return joint.reshape(batch_size, N_ACTIONS)

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for PPO training.

        Args:
            obs: Dict with local_map, self_state, teammate_states, global_info.
            action: (batch, 3) existing action for log_prob computation, or None to sample.
            action_mask: (batch, 324) boolean mask. 1 = legal, 0 = illegal.

        Returns:
            action: (batch, 3) sampled or provided action.
            log_prob: (batch,) total log probability.
            entropy: (batch,) total entropy.
            value: (batch,) state value estimate.
        """
        features = self.get_features(obs)

        move_logits = self.move_head(features)
        skill_logits = self.skill_head(features)
        target_logits = self.target_head(features)

        joint_logits = self._build_joint_logits(move_logits, skill_logits, target_logits)
        if action_mask is not None:
            joint_logits = joint_logits.masked_fill(action_mask <= 0, -1e8)

        dist = torch.distributions.Categorical(logits=joint_logits)

        if action is None:
            flat_action = dist.sample()
            move_action = flat_action // (N_SKILL * N_TARGET)
            skill_action = (flat_action % (N_SKILL * N_TARGET)) // N_TARGET
            target_action = flat_action % N_TARGET
        else:
            move_action = action[:, 0]
            skill_action = action[:, 1]
            target_action = action[:, 2]
            flat_action = move_action * (N_SKILL * N_TARGET) + skill_action * N_TARGET + target_action

        log_prob = dist.log_prob(flat_action)
        entropy = dist.entropy()

        value = self.critic(features).squeeze(-1)

        action_out = torch.stack([move_action, skill_action, target_action], dim=-1)
        return action_out, log_prob, entropy, value

    def get_action(
        self, obs: dict[str, torch.Tensor], action_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Inference-only forward pass — sample actions with mask."""
        with torch.no_grad():
            action, _, _, _ = self.get_action_and_value(obs, action_mask=action_mask)
        return action
