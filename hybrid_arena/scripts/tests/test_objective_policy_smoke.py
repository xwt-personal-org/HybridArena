"""Tests for scripted objective policy smoke verification."""

from hybrid_arena.scripts.objective_policy_smoke import run_objective_policy_smoke


def test_scripted_objective_policy_smoke_reports_base_reachability():
    result = run_objective_policy_smoke(
        episodes=1,
        seed=42,
        map_size=16,
        team_size=2,
        max_steps=500,
    )

    assert result["episodes"] == 1
    assert result["hard_win_rate"] == 1.0
    assert result["base_exposed_rate"] == 1.0
    assert result["avg_base_damage"] > 0.0
    assert result["tower_damage"] > 0.0
    assert result["conclusion"] == "通过"
