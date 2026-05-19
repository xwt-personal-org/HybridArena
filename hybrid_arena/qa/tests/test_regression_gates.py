from hybrid_arena.qa.regression_gates import evaluate_regression_gates


def test_gate_rejects_reward_only_improvement_without_objectives():
    result = evaluate_regression_gates(
        {
            "hard_win_rate": 0.0,
            "base_exposed_rate": 0.0,
            "avg_base_damage": 0.0,
            "avg_tower_damage": 0.0,
            "illegal_action_rate": 0.0,
        },
        reward_improved=True,
    )
    assert not result.passed
    assert "reward-only" in result.failures[0]


def test_gate_passes_when_objective_metric_moves():
    result = evaluate_regression_gates(
        {
            "hard_win_rate": 0.0,
            "base_exposed_rate": 0.0,
            "avg_base_damage": 0.0,
            "avg_tower_damage": 1.0,
            "illegal_action_rate": 0.0,
        },
        reward_improved=True,
    )
    assert result.passed
