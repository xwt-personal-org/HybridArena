"""Evaluation utilities for trained policies.

Supports:
    - Win-rate evaluation vs rule-based / self-play opponents
    - ELO rating tracking
    - KDA and objective statistics
    - Multi-seed aggregation
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from hybrid_arena.algorithms.self_play.elo import ELORatingSystem
from hybrid_arena.minimoba.env import parallel_env


def _call_policy(policy_fn: Callable, obs: dict, agent_id: str):
    try:
        return policy_fn(obs, agent_id)
    except TypeError:
        return policy_fn(obs)


def evaluate_policy(
    policy_fn: Callable,
    opponent_fn: Callable | None = None,
    n_episodes: int = 100,
    env_kwargs: dict | None = None,
    device: str = "cpu",
    seed_offset: int = 0,
) -> dict:
    """Evaluate a policy against an opponent.

    Args:
        policy_fn: Function (obs_dict, agent_id) -> action array.
        opponent_fn: Function (obs_dict, agent_id) -> action array.
                     If None, opponent also uses policy_fn (self-play).
        n_episodes: Number of evaluation episodes.
        env_kwargs: Optional kwargs for parallel_env().
        device: Device for tensor operations.
        seed_offset: Offset for eval seeds.

    Returns:
        Dict with win_rate, avg_reward, avg_length, kda, etc.
    """
    env_kwargs = env_kwargs or {}
    env = parallel_env(**env_kwargs)

    red_wins = 0
    blue_wins = 0
    draws = 0
    hard_red_wins = 0
    hard_blue_wins = 0
    timeout_red_wins = 0
    timeout_blue_wins = 0
    timeout_draws = 0
    episode_red_rewards = []
    episode_blue_rewards = []
    episode_lengths = []
    total_kills = {"red": 0, "blue": 0}
    total_deaths = {"red": 0, "blue": 0}
    towers_destroyed = []
    tower_hp_advantages = []
    tower_damages = []
    base_damages = []
    enemy_base_hp_remaining = []
    base_exposed_count = 0
    start_time = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_red_reward = 0.0
        ep_blue_reward = 0.0
        steps = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent.startswith("red"):
                    actions[agent] = _call_policy(policy_fn, obs[agent], agent)
                else:
                    if opponent_fn is not None:
                        actions[agent] = _call_policy(opponent_fn, obs[agent], agent)
                    else:
                        actions[agent] = _call_policy(policy_fn, obs[agent], agent)

            obs, rewards, terms, truncs, infos = env.step(actions)
            for agent_id, r in rewards.items():
                if agent_id.startswith("red"):
                    ep_red_reward += r
                else:
                    ep_blue_reward += r
            steps += 1

            if any(terms.values()) or any(truncs.values()):
                break

        episode_red_rewards.append(ep_red_reward)
        episode_blue_rewards.append(ep_blue_reward)
        episode_lengths.append(steps)

        if env.is_game_over:
            winner = env.game_state.get_winner() if env.game_state else None
            terminal_reason = (
                getattr(env.game_state, "terminal_reason", None) if env.game_state else None
            )
            if winner == "red":
                red_wins += 1
                if terminal_reason == "base_destroyed":
                    hard_red_wins += 1
                else:
                    timeout_red_wins += 1
            elif winner == "blue":
                blue_wins += 1
                if terminal_reason == "base_destroyed":
                    hard_blue_wins += 1
                else:
                    timeout_blue_wins += 1
            else:
                draws += 1
                if terminal_reason == "timeout":
                    timeout_draws += 1

            total_kills["red"] += getattr(env.game_state, "red_kills", 0)
            total_kills["blue"] += getattr(env.game_state, "blue_kills", 0)
            total_deaths["red"] += sum(
                hero.deaths for hero in env.game_state.heroes.values() if hero.team == "red"
            )
            total_deaths["blue"] += sum(
                hero.deaths for hero in env.game_state.heroes.values() if hero.team == "blue"
            )
            towers_destroyed.append(max(0, 2 - getattr(env.game_state, "blue_towers", 2)))
            red_tower_hp = env.game_state._structure_hp_sum("red", "tower")
            blue_tower_hp = env.game_state._structure_hp_sum("blue", "tower")
            tower_hp_advantages.append(red_tower_hp - blue_tower_hp)

            # Objective shaping diagnostics (Phase F13)
            tower_damages.append(getattr(env.game_state, "red_tower_damage", 0.0))
            base_damages.append(getattr(env.game_state, "red_base_damage", 0.0))
            blue_base_hp = env.game_state._structure_hp_sum("blue", "base")
            enemy_base_hp_remaining.append(blue_base_hp)
            if getattr(env.game_state, "blue_towers", 2) == 0:
                base_exposed_count += 1

    total_games = red_wins + blue_wins + draws
    win_rate = red_wins / total_games if total_games > 0 else 0.0
    elapsed = time.time() - start_time
    total_steps = sum(episode_lengths)

    avg_red_reward = float(np.mean(episode_red_rewards)) if episode_red_rewards else 0.0
    avg_blue_reward = float(np.mean(episode_blue_rewards)) if episode_blue_rewards else 0.0

    return {
        "episodes": total_games,
        "win_rate": win_rate,
        "draw_rate": draws / total_games if total_games > 0 else 0.0,
        "hard_win_rate": hard_red_wins / total_games if total_games > 0 else 0.0,
        "timeout_win_rate": timeout_red_wins / total_games if total_games > 0 else 0.0,
        "timeout_draw_rate": timeout_draws / total_games if total_games > 0 else 0.0,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "hard_red_wins": hard_red_wins,
        "hard_blue_wins": hard_blue_wins,
        "timeout_red_wins": timeout_red_wins,
        "timeout_blue_wins": timeout_blue_wins,
        "timeout_draws": timeout_draws,
        "avg_reward": avg_red_reward,
        "avg_red_reward": avg_red_reward,
        "avg_blue_reward": avg_blue_reward,
        "avg_reward_margin": avg_red_reward - avg_blue_reward,
        "avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "avg_kills": float(total_kills["red"] / max(n_episodes, 1)),
        "avg_deaths": float(total_deaths["red"] / max(n_episodes, 1)),
        "avg_towers_destroyed": float(np.mean(towers_destroyed)) if towers_destroyed else 0.0,
        "avg_tower_hp_advantage": float(np.mean(tower_hp_advantages))
        if tower_hp_advantages
        else 0.0,
        "avg_tower_damage": float(np.mean(tower_damages)) if tower_damages else 0.0,
        "avg_base_damage": float(np.mean(base_damages)) if base_damages else 0.0,
        "avg_enemy_base_hp_remaining": float(np.mean(enemy_base_hp_remaining))
        if enemy_base_hp_remaining
        else 0.0,
        "base_exposed_rate": base_exposed_count / total_games if total_games > 0 else 0.0,
        "fps": total_steps / max(elapsed, 1e-6),
        "total_kills_red": total_kills["red"],
        "total_kills_blue": total_kills["blue"],
    }


def evaluate_against_pool(
    policy_fn: Callable,
    opponent_pool: list,
    n_games_per_opponent: int = 10,
    env_kwargs: dict | None = None,
) -> dict:
    """Evaluate policy against a pool of historical checkpoints.

    Returns aggregate stats and per-opponent win rates.
    """
    results = []
    for opp in opponent_pool:
        # opp is expected to be a callable or state dict
        opp_fn = opp if callable(opp) else lambda obs, aid: obs  # placeholder
        res = evaluate_policy(
            policy_fn,
            opponent_fn=opp_fn,
            n_episodes=n_games_per_opponent,
            env_kwargs=env_kwargs,
        )
        results.append(res)

    avg_win_rate = np.mean([r["win_rate"] for r in results])
    return {
        "avg_win_rate": avg_win_rate,
        "per_opponent": results,
    }


class Evaluator:
    """High-level evaluator that tracks metrics over training."""

    def __init__(
        self,
        env_kwargs: dict | None = None,
        n_eval_episodes: int = 50,
        eval_interval: int = 30_000,
    ):
        self.env_kwargs = env_kwargs or {}
        self.n_eval_episodes = n_eval_episodes
        self.eval_interval = eval_interval
        self.history: list[dict] = []
        self.elo = ELORatingSystem()

    def evaluate(
        self,
        policy_fn: Callable,
        opponent_fn: Callable | None = None,
        opponent_policy: Callable | None = None,
        n_episodes: int | None = None,
        seeds: list[int] | None = None,
        global_step: int = 0,
    ) -> dict:
        """Run evaluation and record results."""
        episodes = n_episodes or self.n_eval_episodes
        seed_offset = seeds[0] if seeds else 0
        result = evaluate_policy(
            policy_fn,
            opponent_fn=opponent_policy or opponent_fn,
            n_episodes=episodes,
            env_kwargs=self.env_kwargs,
            seed_offset=seed_offset,
        )
        result["global_step"] = global_step
        self.history.append(result)
        return result

    def get_best_win_rate(self) -> float:
        if not self.history:
            return 0.0
        return max(r["win_rate"] for r in self.history)

    def summary(self) -> dict:
        if not self.history:
            return {}
        latest = self.history[-1]
        return {
            "latest_win_rate": latest["win_rate"],
            "best_win_rate": self.get_best_win_rate(),
            "n_evals": len(self.history),
            "avg_length": latest.get("avg_length", 0.0),
        }
