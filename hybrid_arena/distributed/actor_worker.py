"""Local actor worker used by the distributed smoke run."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from hybrid_arena.distributed.messages import PolicyVersion, TrajectoryChunk, TrajectoryStep
from hybrid_arena.distributed.policy_sync import PolicyStore
from hybrid_arena.minimoba.action_encoding import decode_action
from hybrid_arena.minimoba.env import parallel_env


class LocalActorWorker:
    def __init__(self, actor_id: str, policy_store: PolicyStore, *, seed: int = 0):
        self.actor_id = actor_id
        self.policy_store = policy_store
        self.rng = np.random.default_rng(seed)
        self.last_fps = 0.0

    def collect(self, steps: int, env_kwargs: dict | None = None) -> TrajectoryChunk:
        env_kwargs = env_kwargs or {"map_size": 16, "team_size": 2, "max_steps": steps}
        policy_version = self.policy_store.current
        env = parallel_env(**env_kwargs)
        obs, _ = env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        collected: list[TrajectoryStep] = []
        start = time.perf_counter()

        while env.agents and len(collected) < steps:
            actions = {}
            for agent_id in env.agents:
                mask = obs[agent_id]["action_mask"]
                valid = np.flatnonzero(mask)
                flat = int(valid[int(self.rng.integers(0, len(valid)))])
                actions[agent_id] = np.asarray(decode_action(flat), dtype=np.int64)

            next_obs, rewards, terms, truncs, _infos = env.step(actions)
            for agent_id, action in actions.items():
                digest = hashlib.sha1(obs[agent_id]["self_state"].tobytes()).hexdigest()[:12]
                collected.append(
                    TrajectoryStep(
                        agent_id=agent_id,
                        action=[int(x) for x in action],
                        reward=float(rewards.get(agent_id, 0.0)),
                        done=bool(terms.get(agent_id, False) or truncs.get(agent_id, False)),
                        behavior_log_prob=0.0,
                        behavior_version=policy_version.version,
                        observation_digest=digest,
                    )
                )
                if len(collected) >= steps:
                    break
            obs = next_obs

        elapsed = max(time.perf_counter() - start, 1e-9)
        self.last_fps = len(collected) / elapsed
        return TrajectoryChunk(
            actor_id=self.actor_id,
            policy_version=PolicyVersion(
                version=policy_version.version,
                updated_at=policy_version.updated_at,
                metadata=policy_version.metadata,
            ),
            steps=collected,
        )
