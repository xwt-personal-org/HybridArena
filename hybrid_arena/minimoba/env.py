"""PettingZoo ParallelEnv wrapper for MiniMOBA-4v4."""

from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from hybrid_arena.minimoba.game_engine import GameState


class MiniMOBAEnv(ParallelEnv):
    """PettingZoo Parallel API environment for 4v4 MOBA with fog of war.

    Action space: MultiDiscrete([9, 4, 9]) — move(9) × skill(4) × target(9) = 324.
    Observation space: Dict with local_map, self_state, teammate_states, global_info, action_mask.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "minimoba_v1",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        map_size: int = 32,
        team_size: int = 4,
        hero_assignments: dict[str, str] | None = None,
        reward_config=None,
        fog_of_war: bool = True,
        max_steps: int = 1000,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.map_size = map_size
        self.team_size = team_size
        self.fog_of_war = fog_of_war
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._seed = seed

        self.possible_agents = [f"red_{i}" for i in range(team_size)] + [
            f"blue_{i}" for i in range(team_size)
        ]
        self.agents: list[str] = self.possible_agents[:]

        self.hero_assignments = hero_assignments
        self.reward_config = reward_config

        self.game_state: GameState | None = None
        self._game_over_rewarded = False

    @functools.cache
    def observation_space(self, agent) -> spaces.Dict:
        return spaces.Dict(
            {
                "local_map": spaces.Box(0.0, 1.0, shape=(11, 11, 11), dtype=np.float32),
                "self_state": spaces.Box(-1.0, 1.0, shape=(20,), dtype=np.float32),
                "teammate_states": spaces.Box(-1.0, 1.0, shape=(3, 15), dtype=np.float32),
                "global_info": spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(324),
            }
        )

    @functools.cache
    def action_space(self, agent) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([9, 4, 9])

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = self.possible_agents[:]
        self.game_state = GameState(
            map_size=self.map_size,
            team_size=self.team_size,
            hero_assignments=self.hero_assignments,
            reward_config=self.reward_config,
            fog_of_war=self.fog_of_war,
            max_steps=self.max_steps,
            seed=seed if seed is not None else self._seed,
        )
        self.game_state.reset(seed=seed if seed is not None else self._seed)
        self._game_over_rewarded = False

        observations = {agent: self.game_state.get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if self.game_state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        step_rewards = self.game_state.step(actions)

        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)

        if self.game_state.is_game_over() and not self._game_over_rewarded:
            self._game_over_rewarded = True
            winner = self.game_state.get_winner()
            for agent in self.agents:
                team = "red" if agent.startswith("red") else "blue"
                if team == winner:
                    step_rewards[agent] += self.game_state.reward_config.win
                else:
                    step_rewards[agent] += self.game_state.reward_config.lose
                terminations[agent] = True

        if self.game_state.step_count >= self.max_steps:
            for agent in self.agents:
                if not terminations[agent]:
                    truncations[agent] = True

        observations = {}
        infos = {}
        for agent in self.agents:
            observations[agent] = self.game_state.get_observation(agent)
            infos[agent] = {
                "episode_step": self.game_state.step_count,
                "red_kills": self.game_state.red_kills,
                "blue_kills": self.game_state.blue_kills,
                "red_towers": self.game_state.red_towers,
                "blue_towers": self.game_state.blue_towers,
            }
            hero = self.game_state.heroes.get(agent)
            if hero:
                infos[agent]["hp_ratio"] = hero.hp_ratio
                infos[agent]["alive"] = hero.alive
                infos[agent].update(self.game_state.get_objective_info(hero.team))

        # Remove terminated/truncated agents
        self.agents = [
            a
            for a in self.agents
            if not terminations.get(a, False) and not truncations.get(a, False)
        ]

        return observations, step_rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None
        from hybrid_arena.minimoba.renderer import Renderer

        renderer = Renderer(self.map_size)
        if self.render_mode == "rgb_array":
            return renderer.render_rgb(self.game_state)
        elif self.render_mode == "human":
            renderer.render_human(self.game_state)

    @property
    def is_game_over(self) -> bool:
        """Convenience property to check if the game has ended."""
        return self.game_state is not None and self.game_state.is_game_over()

    def close(self):
        try:
            import pygame
            pygame.quit()
        except ImportError:
            pass


def parallel_env(**kwargs):
    """Create a parallel MiniMOBA environment."""
    return MiniMOBAEnv(**kwargs)


def raw_env(**kwargs):
    """Create an AEC-wrapped MiniMOBA environment via SuperSuit."""
    from pettingzoo.utils.conversions import parallel_to_aec

    env = MiniMOBAEnv(**kwargs)
    env = parallel_to_aec(env)
    return env
