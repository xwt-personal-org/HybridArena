"""Tests for tactical skill outcome statistics."""

from __future__ import annotations

from hybrid_arena.minimoba.tactical_runtime.schema import (
    GameForwardModel,
    GameSkill,
    GameTrigger,
)
from hybrid_arena.minimoba.tactical_runtime.skill_stats import (
    SkillOutcomeStats,
    apply_stats_to_skill,
    update_stats,
)


def _make_skill(prior=0.5, no_go_traces=0):
    return GameSkill(
        id="farm_resources",
        name="Farm Resources",
        triggers=(GameTrigger(kind="annotation_query", spec="any:resource_soon"),),
        salience=0.7,
        no_go_traces=no_go_traces,
        prior=prior,
        forward_model=GameForwardModel(expected_artifacts=frozenset({"gold_change"})),
        precision=0.9,
        cost_estimate=0.2,
        controller="farm",
        provenance="deterministic_v1",
    )


class TestSkillOutcomeStats:
    """Tests for outcome aggregation."""

    def test_update_stats_tracks_attempts_mean_and_failure_streak(self):
        stats = SkillOutcomeStats(skill_id="farm_resources")

        update_stats(stats, success=False, reward_delta=-1.0)
        update_stats(stats, success=False, reward_delta=0.0)
        update_stats(stats, success=True, reward_delta=2.0)

        assert stats.attempts == 3
        assert stats.successes == 1
        assert stats.failures == 2
        assert stats.mean_reward_delta == 1.0 / 3.0
        assert stats.failure_streak == 0

    def test_apply_stats_increases_prior_for_positive_mean_reward(self):
        skill = _make_skill(prior=0.5)
        stats = SkillOutcomeStats(skill_id="farm_resources")
        update_stats(stats, success=True, reward_delta=2.0)

        updated = apply_stats_to_skill(skill, stats)

        assert updated.prior > skill.prior
        assert updated.no_go_traces == skill.no_go_traces

    def test_apply_stats_decreases_prior_and_increments_no_go_on_failure_streak(self):
        skill = _make_skill(prior=0.5, no_go_traces=2)
        stats = SkillOutcomeStats(skill_id="farm_resources")
        update_stats(stats, success=False, reward_delta=-0.5)
        update_stats(stats, success=False, reward_delta=-0.25)
        update_stats(stats, success=False, reward_delta=-0.25)

        updated = apply_stats_to_skill(skill, stats)

        assert updated.prior < skill.prior
        assert updated.no_go_traces == 3

    def test_apply_stats_clamps_prior(self):
        positive = SkillOutcomeStats(skill_id="farm_resources")
        update_stats(positive, success=True, reward_delta=100.0)
        negative = SkillOutcomeStats(skill_id="farm_resources")
        for _ in range(5):
            update_stats(negative, success=False, reward_delta=-100.0)

        assert apply_stats_to_skill(_make_skill(prior=0.94), positive).prior == 0.95
        assert apply_stats_to_skill(_make_skill(prior=0.06), negative).prior == 0.05

    def test_apply_stats_preserves_skill_identity_and_controller_metadata(self):
        skill = _make_skill()
        stats = SkillOutcomeStats(skill_id="farm_resources")
        update_stats(stats, success=True, reward_delta=1.0)

        updated = apply_stats_to_skill(skill, stats)

        assert updated.id == skill.id
        assert updated.name == skill.name
        assert updated.triggers == skill.triggers
        assert updated.controller == skill.controller
        assert updated.provenance == skill.provenance

