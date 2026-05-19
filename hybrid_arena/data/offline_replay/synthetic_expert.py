"""Synthetic expert replay generation.

These replays come from deterministic rules and are not human or production
gameplay data.
"""

from __future__ import annotations

import numpy as np

from hybrid_arena.minimoba.action_encoding import decode_action, encode_action
from hybrid_arena.minimoba.ctde_state import build_global_state
from hybrid_arena.minimoba.env import parallel_env
from hybrid_arena.minimoba.replay_schema import OfflineTransition


def _legalize(action: np.ndarray, mask: np.ndarray) -> list[int]:
    move, skill, target = [int(x) for x in np.asarray(action, dtype=np.int64)]
    flat = encode_action(move % 9, skill % 4, target % 9)
    if mask[flat] > 0:
        return [move % 9, skill % 4, target % 9]
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return [0, 3, 8]
    return list(decode_action(int(valid[0])))


def deterministic_expert_action(obs: dict, agent_id: str) -> np.ndarray:
    """Return a legal-mask-aware scripted action with mild objective pressure."""
    hp_ratio = float(obs["self_state"][0])
    if hp_ratio < 0.25:
        preferred = np.array([8 if agent_id.startswith("red") else 4, 3, 8], dtype=np.int64)
    elif np.any(obs["local_map"][:, :, 8] > 0.5):
        preferred = np.array([3 if agent_id.startswith("red") else 7, 0, 0], dtype=np.int64)
    else:
        preferred = np.array([3 if agent_id.startswith("red") else 7, 0, 8], dtype=np.int64)
    return np.asarray(_legalize(preferred, obs["action_mask"]), dtype=np.int64)


def generate_synthetic_expert_replay(
    *,
    episodes: int = 2,
    steps_per_episode: int = 8,
    seed: int = 7,
    env_kwargs: dict | None = None,
) -> list[OfflineTransition]:
    env_kwargs = env_kwargs or {"map_size": 16, "team_size": 2, "max_steps": steps_per_episode}
    transitions: list[OfflineTransition] = []
    for episode in range(episodes):
        env = parallel_env(**env_kwargs)
        obs, _ = env.reset(seed=seed + episode)
        episode_id = f"synthetic-{seed}-{episode}"
        for step in range(steps_per_episode):
            assert env.game_state is not None
            global_state = build_global_state(env.game_state).tolist()
            actions = {
                agent_id: deterministic_expert_action(obs[agent_id], agent_id)
                for agent_id in env.agents
            }
            prev_obs = {agent_id: obs[agent_id] for agent_id in env.agents}
            next_obs, rewards, terms, truncs, _infos = env.step(actions)
            next_global_state = build_global_state(env.game_state).tolist()
            done_by_agent = {
                agent_id: bool(terms.get(agent_id, False) or truncs.get(agent_id, False))
                for agent_id in prev_obs
            }
            for agent_id, agent_obs in prev_obs.items():
                transitions.append(
                    OfflineTransition(
                        episode_id=episode_id,
                        step=step,
                        agent_id=agent_id,
                        observation=agent_obs,
                        action=[int(x) for x in actions[agent_id]],
                        reward=float(rewards.get(agent_id, 0.0)),
                        next_observation=next_obs.get(agent_id, agent_obs),
                        done=done_by_agent[agent_id],
                        action_mask=agent_obs["action_mask"].astype(int).tolist(),
                        global_state=global_state,
                        next_global_state=next_global_state,
                        metadata={"source": "deterministic_rule_synthetic"},
                    )
                )
            obs = next_obs
            if not env.agents:
                break
    return transitions
