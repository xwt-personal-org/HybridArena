"""MAPPO: Multi-Agent PPO with CTDE centralized critic."""

from hybrid_arena.algorithms.mappo.config import MAPPOConfig
from hybrid_arena.algorithms.mappo.mappo import MAPPO, CentralizedCritic

__all__ = ["MAPPO", "CentralizedCritic", "MAPPOConfig"]
