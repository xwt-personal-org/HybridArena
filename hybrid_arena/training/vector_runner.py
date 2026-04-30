"""Synchronous runner for multiple PettingZoo parallel MiniMOBA envs."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_arena.minimoba.env import parallel_env


@dataclass(frozen=True)
class AgentRef:
    env_idx: int
    agent_id: str


class SyncParallelEnvRunner:
    """Run multiple MiniMOBA ParallelEnv instances synchronously."""

    def __init__(self, num_envs: int, env_kwargs: dict | None = None, seed: int = 0):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs or {}
        self.seed = seed
        self.envs = [
            parallel_env(**{**self.env_kwargs, "seed": seed + env_idx})
            for env_idx in range(num_envs)
        ]
        self.agent_refs = [
            AgentRef(env_idx, agent_id)
            for env_idx, env in enumerate(self.envs)
            for agent_id in env.possible_agents
        ]
        self.possible_agents = [self._flat_agent(ref) for ref in self.agent_refs]
        self.agents = self.possible_agents[:]

    @property
    def agents_per_env(self) -> int:
        return len(self.envs[0].possible_agents)

    @property
    def game_state(self):
        return self.envs[0].game_state

    def _flat_agent(self, ref: AgentRef) -> str:
        return f"env{ref.env_idx}:{ref.agent_id}"

    def _split_agent(self, flat_agent: str) -> AgentRef:
        env_part, agent_id = flat_agent.split(":", 1)
        return AgentRef(int(env_part.removeprefix("env")), agent_id)

    def reset(self, seed: int | None = None):
        observations = {}
        infos = {}
        for env_idx, env in enumerate(self.envs):
            env_seed = (self.seed if seed is None else seed) + env_idx
            env_obs, env_infos = env.reset(seed=env_seed)
            for agent_id in env.possible_agents:
                flat_agent = f"env{env_idx}:{agent_id}"
                observations[flat_agent] = env_obs[agent_id]
                infos[flat_agent] = env_infos.get(agent_id, {})
        self.agents = self.possible_agents[:]
        return observations, infos

    def step(self, actions):
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for env_idx, env in enumerate(self.envs):
            env_actions = {}
            for flat_agent, action in actions.items():
                ref = self._split_agent(flat_agent)
                if ref.env_idx == env_idx:
                    env_actions[ref.agent_id] = action

            env_obs, env_rewards, env_terms, env_truncs, env_infos = env.step(env_actions)
            done = any(env_terms.values()) or any(env_truncs.values()) or not env.agents
            if done:
                reset_obs, reset_infos = env.reset(seed=self.seed + env_idx)
            else:
                reset_obs, reset_infos = env_obs, env_infos

            for agent_id in env.possible_agents:
                flat_agent = f"env{env_idx}:{agent_id}"
                observations[flat_agent] = reset_obs[agent_id]
                rewards[flat_agent] = env_rewards.get(agent_id, 0.0)
                terminations[flat_agent] = env_terms.get(agent_id, False)
                truncations[flat_agent] = env_truncs.get(agent_id, False)
                infos[flat_agent] = reset_infos.get(agent_id, {})

        self.agents = self.possible_agents[:]
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
