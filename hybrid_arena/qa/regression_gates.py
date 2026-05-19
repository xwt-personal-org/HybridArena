"""Objective-aware QA regression gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

OBJECTIVE_FIELDS = ("hard_win_rate", "base_exposed_rate", "avg_base_damage", "avg_tower_damage")


@dataclass(frozen=True)
class RegressionGateResult:
    passed: bool
    failures: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_regression_gates(metrics: dict, *, reward_improved: bool = False) -> RegressionGateResult:
    failures: list[str] = []
    for field in OBJECTIVE_FIELDS:
        if field not in metrics:
            failures.append(f"missing objective metric: {field}")

    if reward_improved and not any(float(metrics.get(field, 0.0)) > 0.0 for field in OBJECTIVE_FIELDS):
        failures.append("reward-only improvement is not accepted when objective metrics remain zero")

    illegal_action_rate = float(metrics.get("illegal_action_rate", 0.0))
    if illegal_action_rate > 0.0:
        failures.append("illegal_action_rate must remain zero in smoke QA")

    return RegressionGateResult(passed=not failures, failures=failures)
