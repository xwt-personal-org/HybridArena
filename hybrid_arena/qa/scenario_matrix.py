"""Tournament scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TournamentScenario:
    name: str
    policy_name: str
    opponent_name: str
    macro_planner_enabled: bool = False
    env_kwargs: dict | None = None


def default_scenarios() -> list[TournamentScenario]:
    return [
        TournamentScenario(
            name="current_policy_vs_rule_bot",
            policy_name="rule_based_current",
            opponent_name="random_baseline",
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 60},
        ),
        TournamentScenario(
            name="macro_planner_enabled_vs_disabled",
            policy_name="rule_macro_enabled",
            opponent_name="rule_macro_disabled",
            macro_planner_enabled=True,
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 60},
        ),
        TournamentScenario(
            name="objective_stress_test",
            policy_name="rule_objective_pressure",
            opponent_name="random_baseline",
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 80},
        ),
    ]
