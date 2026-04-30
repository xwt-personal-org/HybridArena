"""COMA: Counterfactual Multi-Agent Policy Gradients."""

from hybrid_arena.algorithms.coma.coma import COMA, COMACritic
from hybrid_arena.algorithms.coma.config import COMAConfig

__all__ = ["COMA", "COMACritic", "COMAConfig"]
