import pytest
import torch

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.qa.balance_report import write_reports
from hybrid_arena.qa.scenario_matrix import TournamentScenario, default_scenarios
from hybrid_arena.qa.tournament import make_policy_runner, run_tournament


def test_tournament_returns_required_objective_metrics(tmp_path):
    result = run_tournament(
        episodes=1,
        seed=7,
        scenarios=[
            TournamentScenario(
                name="smoke",
                policy_name="rule",
                opponent_name="random_baseline",
                env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 20},
            )
        ],
    )
    row = result["rows"][0]
    assert row["policy_source"] == "rule_based"
    assert row["evaluation_mode"] == "smoke"
    for field in ("hard_win_rate", "base_exposed_rate", "avg_base_damage", "avg_tower_damage"):
        assert field in row["metrics"]
    assert row["metrics"]["illegal_action_rate_source"] == "pre_step_action_mask"
    assert row["metrics"]["planner_override_rate_source"] == "planner_disabled"
    paths = write_reports(result, tmp_path)
    assert paths["json"].endswith("qa_tournament.json")
    assert "Claim Boundary" in (tmp_path / "qa_tournament.md").read_text(encoding="utf-8")


def test_default_scenarios_use_truthful_smoke_names():
    names = [scenario.name for scenario in default_scenarios()]
    assert "rule_policy_vs_random_smoke" in names
    assert "macro_adapter_smoke_vs_rule" in names
    assert "current_policy_vs_rule_bot" not in names
    assert "macro_planner_enabled_vs_disabled" not in names


def test_checkpoint_policy_requires_artifact():
    with pytest.raises(ValueError, match="checkpoint_path"):
        make_policy_runner("checkpoint")


def test_checkpoint_tournament_reports_artifact(tmp_path):
    checkpoint = tmp_path / "fixture-policy.pt"
    torch.save({"model_state_dict": ActorCritic().state_dict()}, checkpoint)
    result = run_tournament(
        episodes=1,
        seed=7,
        scenarios=[
            TournamentScenario(
                name="checkpoint_smoke",
                policy_name="checkpoint_policy",
                opponent_name="random_baseline",
                env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 10},
                policy_source="checkpoint",
                policy_artifact=str(checkpoint),
                evaluation_mode="checkpoint_bound",
                claim_boundary="Checkpoint-bound QA only for the supplied policy artifact.",
            )
        ],
    )
    row = result["rows"][0]
    assert result["policy_source"] == "checkpoint"
    assert result["rating_subject"] == "checkpoint_policy"
    assert result["evaluation_mode"] == "checkpoint_bound"
    assert row["policy_source"] == "checkpoint"
    assert row["policy_artifact"] == str(checkpoint)
    assert row["evaluation_mode"] == "checkpoint_bound"


def test_macro_adapter_smoke_reports_measured_sources():
    result = run_tournament(
        episodes=1,
        seed=7,
        scenarios=[default_scenarios()[1]],
    )
    row = result["rows"][0]
    assert row["policy_source"] == "rule_based"
    assert row["planner_source"] == "macro_adapter_smoke"
    assert row["evaluation_mode"] == "smoke"
    assert row["metrics"]["planner_override_rate_source"] == (
        "macro_adapter_decisions_per_policy_decision"
    )
    assert row["metrics"]["adapter_call_count"] > 0
