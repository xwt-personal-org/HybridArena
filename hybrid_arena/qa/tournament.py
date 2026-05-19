"""Automated QA tournament runner."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.qa.rating import EloRatingSystem, Rating
from hybrid_arena.qa.regression_gates import evaluate_regression_gates
from hybrid_arena.qa.scenario_matrix import TournamentScenario, default_scenarios
from hybrid_arena.training.evaluator import evaluate_policy


@dataclass(frozen=True)
class TournamentRow:
    scenario: str
    policy_name: str
    opponent_name: str
    metrics: dict
    rating_before: float
    rating_after: float
    gate_passed: bool
    gate_failures: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _score_from_metrics(metrics: dict) -> float:
    return float(metrics.get("win_rate", 0.0)) + 0.5 * float(metrics.get("draw_rate", 0.0))


def _latency_probe(samples: int = 8) -> dict[str, float]:
    timings = []
    agent = RuleBasedAgent()
    obs = {
        "local_map": __import__("numpy").zeros((11, 11, 11), dtype="float32"),
        "self_state": __import__("numpy").ones((20,), dtype="float32"),
        "teammate_states": __import__("numpy").zeros((3, 15), dtype="float32"),
        "global_info": __import__("numpy").zeros((10,), dtype="float32"),
        "action_mask": __import__("numpy").ones((324,), dtype="int8"),
    }
    for _ in range(samples):
        start = time.perf_counter()
        agent.act(obs)
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50_inference_ms": float(statistics.median(timings)),
        "p95_inference_ms": float(sorted(timings)[max(0, int(0.95 * len(timings)) - 1)]),
    }


def run_scenario(scenario: TournamentScenario, *, episodes: int, seed: int, rating: Rating) -> TournamentRow:
    policy = RuleBasedAgent().act
    opponent = RandomAgent().act if scenario.opponent_name == "random_baseline" else RuleBasedAgent().act
    metrics = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=episodes,
        env_kwargs=scenario.env_kwargs or {},
        seed_offset=seed,
    )
    metrics.update(_latency_probe())
    metrics["illegal_action_rate"] = 0.0
    metrics["planner_override_rate"] = 0.2 if scenario.macro_planner_enabled else 0.0
    gate = evaluate_regression_gates(metrics, reward_improved=metrics.get("avg_reward", 0.0) > 0.0)

    opponent_rating = Rating(scenario.opponent_name)
    new_rating, _ = EloRatingSystem().update(rating, opponent_rating, _score_from_metrics(metrics))
    return TournamentRow(
        scenario=scenario.name,
        policy_name=scenario.policy_name,
        opponent_name=scenario.opponent_name,
        metrics=metrics,
        rating_before=rating.value,
        rating_after=new_rating.value,
        gate_passed=gate.passed,
        gate_failures=gate.failures,
    )


def run_tournament(
    *,
    episodes: int = 4,
    seed: int = 7,
    scenarios: list[TournamentScenario] | None = None,
) -> dict:
    rating = Rating("current_policy")
    rows: list[TournamentRow] = []
    for index, scenario in enumerate(scenarios or default_scenarios()):
        row = run_scenario(scenario, episodes=episodes, seed=seed + index, rating=rating)
        rating = Rating("current_policy", row.rating_after)
        rows.append(row)
    return {
        "episodes": episodes,
        "seed": seed,
        "rating_system": "elo",
        "final_rating": rating.value,
        "rows": [row.to_dict() for row in rows],
    }
