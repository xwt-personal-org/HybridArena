"""V-trace return computation for off-policy actor/learner smoke tests."""

from __future__ import annotations

import numpy as np


def vtrace_returns(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    *,
    clip_rho: float = 1.0,
    clip_pg_rho: float = 1.0,
):
    """Compute V-trace targets and policy-gradient advantages."""
    log_rhos = np.asarray(log_rhos, dtype=np.float32)
    discounts = np.asarray(discounts, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    bootstrap_value = np.asarray(bootstrap_value, dtype=np.float32)

    if not (log_rhos.shape == discounts.shape == rewards.shape == values.shape):
        raise ValueError("log_rhos, discounts, rewards, and values must have matching shape")

    rhos = np.exp(log_rhos)
    clipped_rhos = np.minimum(rhos, clip_rho)
    cs = np.minimum(rhos, 1.0)
    values_t_plus_1 = np.concatenate([values[1:], bootstrap_value.reshape(1)])
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    acc = np.float32(0.0)
    vs = np.zeros_like(values, dtype=np.float32)
    for index in range(values.shape[0] - 1, -1, -1):
        acc = deltas[index] + discounts[index] * cs[index] * acc
        vs[index] = values[index] + acc

    vs_t_plus_1 = np.concatenate([vs[1:], bootstrap_value.reshape(1)])
    clipped_pg_rhos = np.minimum(rhos, clip_pg_rho)
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
    return vs.astype(np.float32), pg_advantages.astype(np.float32)
