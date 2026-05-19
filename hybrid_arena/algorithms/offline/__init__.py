"""Offline RL smoke-test interfaces."""

from hybrid_arena.algorithms.offline.bc import BehaviorCloningTrainer, flatten_observation
from hybrid_arena.algorithms.offline.cql import DiscreteCQLTrainer
from hybrid_arena.algorithms.offline.iql import DiscreteIQLTrainer

__all__ = [
    "BehaviorCloningTrainer",
    "DiscreteCQLTrainer",
    "DiscreteIQLTrainer",
    "flatten_observation",
]
