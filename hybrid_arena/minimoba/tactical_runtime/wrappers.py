"""Opt-in wrappers for tactical observation augmentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from gymnasium import spaces

from hybrid_arena.minimoba.tactical_runtime.observation import build_augmented_observation
from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace


class PheromoneObservationAdapter:
    """Parallel-env adapter that adds local_map_with_pheromones on demand."""

    def __init__(
        self,
        env: object,
        workspace: BattlefieldWorkspace,
        position_provider: Callable[[str, dict], tuple[int, int]] | None = None,
        view_size: int = 11,
    ) -> None:
        self.env = env
        self.workspace = workspace
        self.position_provider = position_provider
        self.view_size = view_size

    def reset(self, *args, **kwargs):
        observations, infos = self.env.reset(*args, **kwargs)
        return self._augment_observations(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            self._augment_observations(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def observation_space(self, agent):
        base_space = self.env.observation_space(agent)
        if not isinstance(base_space, spaces.Dict):
            return base_space

        local_map_space = base_space["local_map"]
        local_shape = local_map_space.shape
        augmented_shape = (
            local_shape[0],
            local_shape[1],
            local_shape[2] + 3,
        )
        adapted_spaces: dict[str, spaces.Space] = dict(base_space.spaces)
        adapted_spaces["local_map_with_pheromones"] = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=augmented_shape,
            dtype=np.float32,
        )
        return spaces.Dict(adapted_spaces)

    def _augment_observations(self, observations: dict[str, dict]) -> dict[str, dict]:
        return {
            agent: build_augmented_observation(
                observation=observation,
                workspace=self.workspace,
                agent_position=self._agent_position(agent, observation),
                view_size=self.view_size,
            )
            for agent, observation in observations.items()
        }

    def _agent_position(self, agent: str, observation: dict) -> tuple[int, int]:
        if self.position_provider is not None:
            return self.position_provider(agent, observation)

        game_state = getattr(self.env, "game_state", None)
        hero = getattr(game_state, "heroes", {}).get(agent) if game_state else None
        if hero is not None:
            return int(getattr(hero, "x", 0)), int(getattr(hero, "y", 0))
        return (0, 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

