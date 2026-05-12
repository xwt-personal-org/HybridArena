"""Tests for opt-in tactical observation wrappers."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from hybrid_arena.minimoba.tactical_runtime.workspace import (
    BattlefieldAnnotation,
    BattlefieldWorkspace,
)
from hybrid_arena.minimoba.tactical_runtime.wrappers import PheromoneObservationAdapter


class _StubParallelEnv:
    """Minimal parallel env for adapter tests."""

    possible_agents = ["red_0"]
    agents = ["red_0"]

    def __init__(self):
        self._obs = {
            "red_0": {
                "local_map": np.zeros((11, 11, 11), dtype=np.float32),
                "self_state": np.zeros((20,), dtype=np.float32),
            }
        }

    def observation_space(self, agent):
        return spaces.Dict({
            "local_map": spaces.Box(0.0, 1.0, shape=(11, 11, 11), dtype=np.float32),
            "self_state": spaces.Box(-1.0, 1.0, shape=(20,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        return self._obs, {"red_0": {}}

    def step(self, actions):
        return self._obs, {"red_0": 0.0}, {"red_0": False}, {"red_0": False}, {"red_0": {}}


class TestPheromoneObservationAdapter:
    """Tests for opt-in observation augmentation."""

    def test_reset_adds_augmented_map_without_mutating_original_observation(self):
        env = _StubParallelEnv()
        workspace = BattlefieldWorkspace(map_size=32)
        workspace.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"dangerous"}))
        adapter = PheromoneObservationAdapter(
            env,
            workspace=workspace,
            position_provider=lambda agent, observation: (5, 5),
        )

        observations, infos = adapter.reset()

        assert infos == {"red_0": {}}
        assert "local_map_with_pheromones" in observations["red_0"]
        assert observations["red_0"]["local_map_with_pheromones"].shape == (11, 11, 14)
        assert "local_map_with_pheromones" not in env._obs["red_0"]
        assert observations["red_0"]["local_map"] is env._obs["red_0"]["local_map"]

    def test_observation_space_is_adapter_local(self):
        env = _StubParallelEnv()
        adapter = PheromoneObservationAdapter(
            env,
            workspace=BattlefieldWorkspace(map_size=32),
            position_provider=lambda agent, observation: (5, 5),
        )

        adapted_space = adapter.observation_space("red_0")
        base_space = env.observation_space("red_0")

        assert adapted_space["local_map"].shape == (11, 11, 11)
        assert adapted_space["local_map_with_pheromones"].shape == (11, 11, 14)
        assert "local_map_with_pheromones" not in base_space

