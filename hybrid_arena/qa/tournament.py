"""Automated QA tournament runner."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hybrid_arena.inference.adapter import MacroActionAdapter
from hybrid_arena.minimoba.action_encoding import encode_action
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
    policy_source: str
    policy_artifact: str | None
    planner_source: str
    evaluation_mode: str
    claim_boundary: str
    open_items: list[str]
    metrics: dict
    rating_before: float
    rating_after: float
    gate_passed: bool
    gate_failures: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyRunner:
    action_fn: Callable
    policy_source: str
    policy_artifact: str | None
    planner_source: str
    evaluation_mode: str
    claim_boundary: str
    decisions: int = 0
    illegal_actions: int = 0
    planner_calls: int = 0
    adapter_calls: int = 0

    def __call__(self, obs: dict, agent_id: str | None = None):
        self.decisions += 1
        if self.planner_source == "macro_adapter_smoke":
            self.planner_calls += 1
            self.adapter_calls += 1
        action = self.action_fn(obs)
        if not _action_is_legal(action, obs):
            self.illegal_actions += 1
        return action

    def metric_sources(self) -> dict:
        total = max(self.decisions, 1)
        planner_active = self.planner_source != "none"
        return {
            "policy_decision_count": self.decisions,
            "planner_call_count": self.planner_calls,
            "adapter_call_count": self.adapter_calls,
            "illegal_action_rate": self.illegal_actions / total,
            "illegal_action_rate_source": "pre_step_action_mask",
            "planner_override_rate": self.adapter_calls / total if planner_active else 0.0,
            "planner_override_rate_source": (
                "macro_adapter_decisions_per_policy_decision"
                if planner_active
                else "planner_disabled"
            ),
        }


def _action_is_legal(action, obs: dict) -> bool:
    mask = obs.get("action_mask")
    if mask is None:
        return False
    try:
        components = np.asarray(action, dtype=np.int64).reshape(-1)
        if components.size != 3:
            return False
        flat = encode_action(int(components[0]), int(components[1]), int(components[2]))
    except (TypeError, ValueError):
        return False
    return bool(mask[flat] > 0)


def _checkpoint_action_fn(checkpoint_path: Path) -> Callable:
    import torch

    from hybrid_arena.algorithms.networks import ActorCritic
    from hybrid_arena.deployment.export_onnx import load_actor_critic_checkpoint

    policy = ActorCritic()
    load_actor_critic_checkpoint(policy, checkpoint_path)
    policy.eval()

    def act(obs: dict):
        tensor_obs = {
            "local_map": torch.as_tensor(obs["local_map"], dtype=torch.float32).unsqueeze(0),
            "self_state": torch.as_tensor(obs["self_state"], dtype=torch.float32).unsqueeze(0),
            "teammate_states": torch.as_tensor(obs["teammate_states"], dtype=torch.float32).unsqueeze(0),
            "global_info": torch.as_tensor(obs["global_info"], dtype=torch.float32).unsqueeze(0),
        }
        action_mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy.get_action(tensor_obs, action_mask=action_mask)
        return action.cpu().numpy()[0].astype(np.int64)

    return act


def make_policy_runner(
    policy_source: str,
    checkpoint_path: str | Path | None = None,
    planner_source: str = "none",
) -> PolicyRunner:
    policy_artifact = str(checkpoint_path) if checkpoint_path is not None else None
    claim_boundary = "Rule-smoke QA only; not current trained-policy validation."
    if planner_source == "macro_adapter_smoke":
        return PolicyRunner(
            action_fn=MacroActionAdapter("PUSH_LANE").act,
            policy_source=policy_source,
            policy_artifact=policy_artifact,
            planner_source=planner_source,
            evaluation_mode="smoke",
            claim_boundary="MacroActionAdapter smoke only; not external LLM gameplay proof.",
        )
    if policy_source == "rule_based":
        return PolicyRunner(
            action_fn=RuleBasedAgent().act,
            policy_source=policy_source,
            policy_artifact=policy_artifact,
            planner_source=planner_source,
            evaluation_mode="smoke",
            claim_boundary=claim_boundary,
        )
    if policy_source == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint policy_source requires checkpoint_path")
        return PolicyRunner(
            action_fn=_checkpoint_action_fn(Path(checkpoint_path)),
            policy_source=policy_source,
            policy_artifact=policy_artifact,
            planner_source=planner_source,
            evaluation_mode="checkpoint_bound",
            claim_boundary="Checkpoint-bound QA only for the supplied policy artifact.",
        )
    raise ValueError(f"unknown policy_source: {policy_source!r}")


def _score_from_metrics(metrics: dict) -> float:
    return float(metrics.get("win_rate", 0.0)) + 0.5 * float(metrics.get("draw_rate", 0.0))


def _latency_probe(samples: int = 8) -> dict[str, float]:
    timings = []
    agent = RuleBasedAgent()
    obs = {
        "local_map": np.zeros((11, 11, 11), dtype="float32"),
        "self_state": np.ones((20,), dtype="float32"),
        "teammate_states": np.zeros((3, 15), dtype="float32"),
        "global_info": np.zeros((10,), dtype="float32"),
        "action_mask": np.ones((324,), dtype="int8"),
    }
    for _ in range(samples):
        start = time.perf_counter()
        agent.act(obs)
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50_inference_ms": float(statistics.median(timings)),
        "p95_inference_ms": float(sorted(timings)[max(0, int(0.95 * len(timings)) - 1)]),
    }


def run_scenario(
    scenario: TournamentScenario,
    *,
    episodes: int,
    seed: int,
    rating: Rating,
) -> TournamentRow:
    policy = make_policy_runner(
        scenario.policy_source,
        checkpoint_path=scenario.policy_artifact,
        planner_source=scenario.planner_source,
    )
    opponent = (
        RandomAgent().act
        if scenario.opponent_name == "random_baseline"
        else RuleBasedAgent().act
    )
    metrics = evaluate_policy(
        policy,
        opponent_fn=opponent,
        n_episodes=episodes,
        env_kwargs=scenario.env_kwargs or {},
        seed_offset=seed,
    )
    metrics.update(_latency_probe())
    metrics.update(policy.metric_sources())
    metrics["policy_source"] = policy.policy_source
    metrics["policy_artifact"] = policy.policy_artifact
    metrics["planner_source"] = policy.planner_source
    metrics["evaluation_mode"] = scenario.evaluation_mode
    metrics["claim_boundary"] = scenario.claim_boundary
    gate = evaluate_regression_gates(metrics, reward_improved=metrics.get("avg_reward", 0.0) > 0.0)

    opponent_rating = Rating(scenario.opponent_name)
    new_rating, _ = EloRatingSystem().update(rating, opponent_rating, _score_from_metrics(metrics))
    return TournamentRow(
        scenario=scenario.name,
        policy_name=scenario.policy_name,
        opponent_name=scenario.opponent_name,
        policy_source=policy.policy_source,
        policy_artifact=policy.policy_artifact,
        planner_source=policy.planner_source,
        evaluation_mode=scenario.evaluation_mode,
        claim_boundary=scenario.claim_boundary,
        open_items=list(scenario.open_items),
        metrics=metrics,
        rating_before=rating.value,
        rating_after=new_rating.value,
        gate_passed=gate.passed,
        gate_failures=gate.failures,
    )


def _single_or_mixed(values: list[str]) -> str:
    return values[0] if len(set(values)) == 1 else "mixed"


def run_tournament(
    *,
    episodes: int = 4,
    seed: int = 7,
    scenarios: list[TournamentScenario] | None = None,
) -> dict:
    active_scenarios = scenarios or default_scenarios()
    rating = Rating("rule_based_smoke")
    rows: list[TournamentRow] = []
    for index, scenario in enumerate(active_scenarios):
        row = run_scenario(scenario, episodes=episodes, seed=seed + index, rating=rating)
        rating = Rating("rule_based_smoke", row.rating_after)
        rows.append(row)
    policy_sources = [row.policy_source for row in rows]
    planner_sources = [row.planner_source for row in rows]
    evaluation_modes = [row.evaluation_mode for row in rows]
    return {
        "episodes": episodes,
        "seed": seed,
        "rating_system": "elo",
        "rating_subject": "rule_based_smoke",
        "policy_source": _single_or_mixed(policy_sources),
        "planner_source": _single_or_mixed(planner_sources),
        "evaluation_mode": _single_or_mixed(evaluation_modes),
        "claim_boundary": "QA reports are smoke evidence unless policy/planner artifacts prove otherwise.",
        "final_rating": rating.value,
        "rows": [row.to_dict() for row in rows],
    }
