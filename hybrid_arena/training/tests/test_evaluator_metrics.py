"""Evaluator result metrics consistency tests."""

import json

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.training.evaluator import evaluate_policy

REQUIRED_FIELDS = [
    "episodes",
    "win_rate",
    "draw_rate",
    "avg_reward",
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
        policy, opponent_fn=opponent, n_episodes=5,
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
        policy, opponent_fn=opponent, n_episodes=3,
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
        policy, opponent_fn=opponent, n_episodes=3,
        env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 30},
        seed_offset=42,
    )
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    for field in REQUIRED_FIELDS:
        assert field in deserialized, f"Missing required field after serialization: {field}"
