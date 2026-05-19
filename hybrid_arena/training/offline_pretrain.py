"""Offline pretraining smoke helpers."""

from __future__ import annotations

from hybrid_arena.algorithms.offline.bc import BehaviorCloningTrainer, flatten_observation
from hybrid_arena.minimoba.action_encoding import encode_action
from hybrid_arena.minimoba.replay_schema import OfflineTransition

OBJECTIVE_METRICS = ("hard_win_rate", "base_exposed_rate", "avg_base_damage", "avg_tower_damage")


def objective_reachability_guard(metrics: dict[str, float]) -> None:
    """Reject reward-only claims when no objective metric moves."""
    if all(float(metrics.get(name, 0.0)) <= 0.0 for name in OBJECTIVE_METRICS):
        raise ValueError(
            "Objective reachability failed: hard win, base exposure, base damage, "
            "and tower damage are all zero."
        )


def behavior_cloning_smoke(transitions: list[OfflineTransition], epochs: int = 8) -> dict[str, float]:
    observations = [flatten_observation(t.observation) for t in transitions]
    actions = [encode_action(*[int(x) for x in t.action]) for t in transitions]
    import numpy as np

    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    trainer = BehaviorCloningTrainer(input_dim=obs_array.shape[1], seed=7)
    initial = trainer.train_epoch(obs_array, action_array)
    latest = initial
    for _ in range(max(epochs - 1, 0)):
        latest = trainer.train_epoch(obs_array, action_array)
    return {"initial_loss": initial, "final_loss": latest}
