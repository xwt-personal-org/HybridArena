"""Evaluator result metrics consistency tests."""

import json

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.training.evaluator import evaluate_policy

REQUIRED_FIELDS = [
    "episodes",
    "win_rate",
    "draw_rate",
    "hard_win_rate",
    "timeout_win_rate",
    "timeout_draw_rate",
    "avg_reward",
    "avg_red_reward",
    "avg_blue_reward",
    "avg_reward_margin",
    "avg_episode_length",
    "avg_kills",
    "avg_deaths",
    "avg_towers_destroyed",
    "avg_tower_hp_advantage",
    "fps",
]


def test_win_rate_denominator_excludes_draws_or_documents_draws():
    """win_rate should be red_wins / total games; draw_rate documents draw fraction."""
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=5,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 30},
        seed_offset=42,
    )
    assert 0.0 <= result["win_rate"] <= 1.0
    assert 0.0 <= result["draw_rate"] <= 1.0
    assert result["win_rate"] + result["draw_rate"] <= 1.0 + 1e-9
    assert "red_wins" in result
    assert "blue_wins" in result
    assert "draws" in result


def test_objective_metrics_present_in_eval_result():
    """All required objective fields must exist in the result dict."""
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=3,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 30},
        seed_offset=42,
    )
    for field in REQUIRED_FIELDS:
        assert field in result, f"Missing required field: {field}"
        assert result[field] is not None, f"Field {field} is None"


def test_eval_result_serialization_contains_required_fields():
    """Result dict must be JSON-serializable with all required fields."""
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=3,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 30},
        seed_offset=42,
    )
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    for field in REQUIRED_FIELDS:
        assert field in deserialized, f"Missing required field after serialization: {field}"


def test_avg_reward_equals_red_reward():
    """avg_reward must equal avg_red_reward (not cross-team average)."""
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=5,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 30},
        seed_offset=42,
    )
    assert result["avg_reward"] == result["avg_red_reward"]
    assert "avg_blue_reward" in result
    assert "avg_reward_margin" in result


def test_reward_margin_is_red_minus_blue():
    """avg_reward_margin should equal avg_red_reward - avg_blue_reward."""
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=5,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 30},
        seed_offset=42,
    )
    expected_margin = result["avg_red_reward"] - result["avg_blue_reward"]
    assert abs(result["avg_reward_margin"] - expected_margin) < 1e-6


def test_red_blue_rewards_not_cross_averaged():
    """Red and blue rewards must be tracked separately, not averaged together.

    If red wins, red gets positive win reward and blue gets negative lose reward.
    The old bug averaged them together, diluting the signal.
    """
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=10,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 100},
        seed_offset=42,
    )
    # If there are wins, red and blue rewards should differ
    if result["red_wins"] > 0 or result["blue_wins"] > 0:
        # Red reward should not equal blue reward when there are wins
        assert result["avg_red_reward"] != result["avg_blue_reward"], (
            "Red and blue rewards should differ when there are wins"
        )
        # Margin should be non-zero
        assert result["avg_reward_margin"] != 0.0, (
            "Reward margin should be non-zero when there are wins"
        )


def test_evaluator_splits_hard_win_and_timeout_win():
    """Evaluator must split wins into hard (base_destroyed) and timeout adjudicated."""
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=10,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 100},
        seed_offset=42,
    )
    # Verify all win types are tracked
    assert "hard_red_wins" in result
    assert "hard_blue_wins" in result
    assert "timeout_red_wins" in result
    assert "timeout_blue_wins" in result
    assert "timeout_draws" in result

    # hard + timeout wins should equal total wins
    assert result["hard_red_wins"] + result["timeout_red_wins"] == result["red_wins"]
    assert result["hard_blue_wins"] + result["timeout_blue_wins"] == result["blue_wins"]

    # hard_win_rate + timeout_win_rate + timeout_draw_rate + other_draws should be <= 1
    total_accounted = (
        result["hard_win_rate"] + result["timeout_win_rate"] + result["timeout_draw_rate"]
    )
    assert total_accounted <= 1.0 + 1e-9


def test_hard_win_rate_is_base_destroyed_only():
    """hard_win_rate should only count base_destroyed wins, not timeout adjudicated."""
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act
    result = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=10,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 100},
        seed_offset=42,
    )
    # hard_win_rate should be <= win_rate (since it's a subset)
    assert result["hard_win_rate"] <= result["win_rate"] + 1e-9
    # timeout_win_rate should be <= win_rate
    assert result["timeout_win_rate"] <= result["win_rate"] + 1e-9
