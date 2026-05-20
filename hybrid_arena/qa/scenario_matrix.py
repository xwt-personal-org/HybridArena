"""Tournament scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass

RULE_SMOKE_BOUNDARY = "Rule-smoke QA only; not current trained-policy validation."
CHECKPOINT_BOUNDARY = (
    "Checkpoint-bound QA for the supplied artifact only; not high-skill policy proof."
)


@dataclass(frozen=True)
class TournamentScenario:
    name: str
    policy_name: str
    opponent_name: str
    macro_planner_enabled: bool = False
    env_kwargs: dict | None = None
    policy_source: str = "rule_based"
    policy_artifact: str | None = None
    planner_source: str = "none"
    evaluation_mode: str = "smoke"
    claim_boundary: str = RULE_SMOKE_BOUNDARY
    open_items: tuple[str, ...] = ()


def default_scenarios() -> list[TournamentScenario]:
    return [
        TournamentScenario(
            name="rule_policy_vs_random_smoke",
            policy_name="rule_based_smoke",
            opponent_name="random_baseline",
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 60},
        ),
        TournamentScenario(
            name="macro_adapter_smoke_vs_rule",
            policy_name="rule_based_macro_adapter_smoke",
            opponent_name="rule_baseline",
            macro_planner_enabled=True,
            planner_source="macro_adapter_smoke",
            claim_boundary="MacroActionAdapter smoke only; not external LLM gameplay proof.",
            open_items=("external LLM planner gameplay validation not run",),
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 60},
        ),
        TournamentScenario(
            name="objective_stress_smoke",
            policy_name="rule_objective_pressure_smoke",
            opponent_name="random_baseline",
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 80},
        ),
    ]


def checkpoint_scenarios(checkpoint_path: str) -> list[TournamentScenario]:
    return [
        TournamentScenario(
            name="checkpoint_policy_vs_random_smoke",
            policy_name="checkpoint_policy_smoke",
            opponent_name="random_baseline",
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 60},
            policy_source="checkpoint",
            policy_artifact=checkpoint_path,
            evaluation_mode="checkpoint_bound",
            claim_boundary=CHECKPOINT_BOUNDARY,
            open_items=("tiny checkpoint evidence chain; strategy quality not established",),
        ),
        TournamentScenario(
            name="checkpoint_objective_stress_smoke",
            policy_name="checkpoint_objective_pressure_smoke",
            opponent_name="random_baseline",
            env_kwargs={"map_size": 16, "team_size": 2, "max_steps": 80},
            policy_source="checkpoint",
            policy_artifact=checkpoint_path,
            evaluation_mode="checkpoint_bound",
            claim_boundary=CHECKPOINT_BOUNDARY,
            open_items=("tiny checkpoint evidence chain; strategy quality not established",),
        ),
    ]
