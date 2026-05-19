from hybrid_arena.qa.balance_report import write_reports
from hybrid_arena.qa.scenario_matrix import TournamentScenario
from hybrid_arena.qa.tournament import run_tournament


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
    for field in ("hard_win_rate", "base_exposed_rate", "avg_base_damage", "avg_tower_damage"):
        assert field in row["metrics"]
    paths = write_reports(result, tmp_path)
    assert paths["json"].endswith("qa_tournament.json")
