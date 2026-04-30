"""MAPPO configuration."""

from dataclasses import dataclass

from hybrid_arena.algorithms.ppo.config import PPOConfig


@dataclass
class MAPPOConfig(PPOConfig):
    """MAPPO hyperparameters (extends PPOConfig with CTDE options)."""

    use_shared_critic: bool = True
    # Critic hidden dim is 2x actor hidden dim by default
    critic_hidden_dim: int | None = None  # None -> auto = hidden_dim * 2

    def __post_init__(self):
        if self.critic_hidden_dim is None:
            self.critic_hidden_dim = self.hidden_dim * 2
