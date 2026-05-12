"""No-gradient tactical skill outcome statistics."""

from __future__ import annotations

from dataclasses import dataclass, replace

from hybrid_arena.minimoba.tactical_runtime.schema import GameSkill


@dataclass
class SkillOutcomeStats:
    """Aggregated outcomes for one tactical skill."""

    skill_id: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_reward_delta: float = 0.0
    failure_streak: int = 0

    @property
    def mean_reward_delta(self) -> float:
        """Mean reward delta across all attempts."""
        if self.attempts == 0:
            return 0.0
        return self.total_reward_delta / self.attempts


def update_stats(
    stats: SkillOutcomeStats,
    success: bool,
    reward_delta: float,
) -> SkillOutcomeStats:
    """Update aggregate stats in place and return the same object."""
    stats.attempts += 1
    stats.total_reward_delta += reward_delta
    if success:
        stats.successes += 1
        stats.failure_streak = 0
    else:
        stats.failures += 1
        stats.failure_streak += 1
    return stats


def apply_stats_to_skill(skill: GameSkill, stats: SkillOutcomeStats) -> GameSkill:
    """Return a skill copy with prior/no-go traces adjusted from outcomes."""
    prior = skill.prior
    no_go_traces = skill.no_go_traces

    if stats.mean_reward_delta > 0:
        prior += min(0.1, max(0.01, stats.mean_reward_delta * 0.05))
    elif stats.failure_streak >= 3:
        prior -= min(0.1, max(0.01, abs(stats.mean_reward_delta) * 0.05))
        no_go_traces += 1

    return replace(
        skill,
        prior=_clamp(prior, 0.05, 0.95),
        no_go_traces=no_go_traces,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)

