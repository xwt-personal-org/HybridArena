"""Environment wrappers for MiniMOBA: single-agent, observation normalization."""

from __future__ import annotations

from gymnasium import Env


def make_single_agent(
    map_size: int = 32,
    team_size: int = 4,
    control_agent: str = "red_0",
    teammate_policy=None,
    opponent_policy=None,
    fog_of_war: bool = True,
    max_steps: int = 1000,
    render_mode: str | None = None,
):
    """Create a Gymnasium-compatible single-agent wrapper.

    The controlled agent acts; all other agents follow teammate_policy / opponent_policy.
    If no policy is provided, random actions are used.

    Returns:
        SingleAgentWrapper instance.
    """
    from hybrid_arena.minimoba.env import MiniMOBAEnv

    env = MiniMOBAEnv(
        map_size=map_size,
        team_size=team_size,
        fog_of_war=fog_of_war,
        max_steps=max_steps,
        render_mode=render_mode,
    )
    return SingleAgentWrapper(
        env,
        control_agent=control_agent,
        teammate_policy=teammate_policy,
        opponent_policy=opponent_policy,
    )


class SingleAgentWrapper(Env):
    """Wraps a ParallelEnv into a single-agent Gymnasium Env."""

    def __init__(
        self,
        parallel_env,
        control_agent: str = "red_0",
        teammate_policy=None,
        opponent_policy=None,
    ):
        self.parallel_env = parallel_env
        self.control_agent = control_agent
        self.teammate_policy = teammate_policy
        self.opponent_policy = opponent_policy
        self._control_team = "red" if control_agent.startswith("red") else "blue"

        self.observation_space = parallel_env.observation_space(control_agent)
        self.action_space = parallel_env.action_space(control_agent)
        self.metadata = parallel_env.metadata
        self.render_mode = parallel_env.render_mode

        self._obs = None
        self._done = False

    def reset(self, seed=None, options=None):
        observations, infos = self.parallel_env.reset(seed=seed, options=options)
        self._obs = observations
        self._done = False
        return observations[self.control_agent], infos.get(self.control_agent, {})

    def step(self, action):
        actions = {}
        for agent in self.parallel_env.agents:
            if agent == self.control_agent:
                actions[agent] = action
            else:
                team = "red" if agent.startswith("red") else "blue"
                policy = self.teammate_policy if team == self._control_team else self.opponent_policy
                if policy is not None:
                    actions[agent] = policy(self._obs[agent])
                else:
                    actions[agent] = self.parallel_env.action_space(agent).sample()

        observations, rewards, terminations, truncations, infos = self.parallel_env.step(actions)
        self._obs = observations

        reward = rewards.get(self.control_agent, 0.0)
        terminated = terminations.get(self.control_agent, False)
        truncated = truncations.get(self.control_agent, False)
        info = infos.get(self.control_agent, {})

        obs = observations.get(self.control_agent, self._empty_obs())
        self._done = terminated or truncated

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.parallel_env.render()

    def close(self):
        self.parallel_env.close()

    def _empty_obs(self):
        return self.observation_space.sample()
