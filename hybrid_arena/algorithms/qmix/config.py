"""QMIX configuration."""

from dataclasses import dataclass


@dataclass
class QMIXConfig:
    """QMIX hyperparameters."""

    # Environment
    map_size: int = 32
    team_size: int = 4
    max_steps: int = 1000
    fog_of_war: bool = True

    # Training
    total_timesteps: int = 3_000_000
    buffer_size: int = 50_000
    batch_size: int = 128
    learning_rate: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.005  # target soft update
    target_update_freq: int = 200  # hard update every N steps (alternative to soft)

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    # Network
    hidden_dim: int = 48
    mixing_embed_dim: int = 32

    # Misc
    seed: int = 42
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
