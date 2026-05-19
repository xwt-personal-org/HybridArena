"""Contracts for centralized training with decentralized execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentralizedCriticInput:
    """Inputs allowed for centralized critics, not decentralized actors."""

    local_observation: dict
    global_state: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True)
class CTDEBatch:
    actor_observations: list[dict]
    critic_global_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def validate_decentralized_actor_inputs(self) -> None:
        for obs in self.actor_observations:
            if "ctde_global_state" in obs or "global_state" in obs:
                raise ValueError("Actor observations must not contain centralized global state.")
