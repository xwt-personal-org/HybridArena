"""Self-play opponent pool + curriculum learning for MiniMOBA."""

from hybrid_arena.algorithms.self_play.curriculum import (
    DEFAULT_LEVELS,
    CurriculumLevel,
    CurriculumScheduler,
)
from hybrid_arena.algorithms.self_play.elo import ELORatingSystem, ELORecord
from hybrid_arena.algorithms.self_play.manager import (
    PolicyCheckpoint,
    SelfPlayManager,
)

__all__ = [
    "ELORatingSystem",
    "ELORecord",
    "PolicyCheckpoint",
    "SelfPlayManager",
    "CurriculumScheduler",
    "CurriculumLevel",
    "DEFAULT_LEVELS",
]
