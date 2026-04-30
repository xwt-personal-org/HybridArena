"""PPO algorithm configuration."""

from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    """PPO hyperparameters compatible with CleanRL style."""

    # Environment
    map_size: int = 32
    team_size: int = 4
    max_steps: int = 1000
    fog_of_war: bool = True

    # Training budget
    total_timesteps: int = 3_000_000
    num_envs: int = 4
    num_steps: int = 128  # rollout steps per env per iteration

    # PPO core
    learning_rate: float = 3e-4
    n_epochs: int = 6
    batch_size: int = 128
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5

    # Dual-clip
    dual_clip_c: float = 3.0  # lower-bound clip for negative advantages

    # Loss coefficients
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    normalize_advantage: bool = True

    # Entropy schedule
    entropy_schedule: str = "linear_decay"  # "linear_decay" | "cosine_decay" | "fixed"
    entropy_start: float = 0.05
    entropy_end: float = 0.001

    # Network
    hidden_dim: int = 48

    # Seed
    seed: int = 42

    # Self-play
    use_self_play: bool = False
    self_play_pool_size: int = 10

    # Evaluation
    eval_interval: int = 30_000

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


@dataclass
class MAPPOPolicy:
    """Minimal placeholder for MAPPO (multi-agent PPO)."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    use_shared_critic: bool = True  # CTDE: centralized critic during training
