"""COMA configuration."""

from dataclasses import dataclass


@dataclass
class COMAConfig:
    """COMA hyperparameters."""

    # Environment
    map_size: int = 32
    team_size: int = 4
    max_steps: int = 1000
    fog_of_war: bool = True

    # Training
    total_timesteps: int = 3_000_000
    gamma: float = 0.99
    td_lambda: float = 0.8
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3

    # Network
    hidden_dim: int = 48

    # Misc
    seed: int = 42
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
