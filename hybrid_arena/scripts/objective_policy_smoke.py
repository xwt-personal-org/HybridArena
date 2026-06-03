"""Scripted objective policy smoke test for MiniMOBA base reachability."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_arena.minimoba.env import parallel_env
from hybrid_arena.minimoba.game_engine import MOVEMENT_DELTA, GameState
from hybrid_arena.minimoba.map_generator import OBSTACLE
from hybrid_arena.minimoba.objectives import StructureState
from hybrid_arena.minimoba.reward_shaper import RewardConfig

CONCLUSION_PASS = "通过"
CONCLUSION_FIX_ENV = "需修环境"
CONCLUSION_FIX_REWARD = "需修 reward"
CONCLUSION_ADJUST_TRAINING = "需调整训练"


def _noop_action() -> np.ndarray:
    return np.array([0, 3, 8], dtype=np.int64)


def _attack_action(move_dir: int = 0) -> np.ndarray:
    return np.array([move_dir, 0, 8], dtype=np.int64)


def _alive_enemy_objectives(game: GameState, team: str) -> list[StructureState]:
    enemy = "blue" if team == "red" else "red"
    towers = [
        structure
        for structure in game.structures.values()
        if structure.team == enemy and structure.structure_type == "tower" and structure.alive
    ]
    if towers:
        return towers
    return [
        structure
        for structure in game.structures.values()
        if structure.team == enemy and structure.structure_type == "base" and structure.alive
    ]


def _nearest_objective(game: GameState, agent_id: str) -> StructureState | None:
    hero = game.heroes[agent_id]
    objectives = _alive_enemy_objectives(game, hero.team)
    if not objectives:
        return None
    return min(objectives, key=lambda obj: abs(hero.x - obj.x) + abs(hero.y - obj.y))


def _next_move_dir(game: GameState, agent_id: str, target: StructureState) -> int:
    hero = game.heroes[agent_id]
    start = (hero.x, hero.y)
    if abs(hero.x - target.x) + abs(hero.y - target.y) <= hero.config.attack_range:
        return 0

    queue: deque[tuple[int, int]] = deque([start])
    first_step: dict[tuple[int, int], int] = {start: 0}
    seen = {start}

    while queue:
        x, y = queue.popleft()
        for move_dir, (dx, dy) in MOVEMENT_DELTA.items():
            if move_dir == 0:
                continue
            nx = x + dx
            ny = y + dy
            pos = (nx, ny)
            if pos in seen:
                continue
            if not (0 <= nx < game.map_size and 0 <= ny < game.map_size):
                continue
            if game.terrain[ny, nx] == OBSTACLE:
                continue

            seen.add(pos)
            first_step[pos] = move_dir if (x, y) == start else first_step[(x, y)]
            if abs(nx - target.x) + abs(ny - target.y) <= hero.config.attack_range:
                return first_step[pos]
            queue.append(pos)

    return 0


def _scripted_objective_action(game: GameState, agent_id: str, attacker_id: str) -> np.ndarray:
    if agent_id != attacker_id:
        return _noop_action()

    target = _nearest_objective(game, agent_id)
    if target is None:
        return _noop_action()

    move_dir = _next_move_dir(game, agent_id, target)
    return _attack_action(move_dir)


def classify_objective_policy_smoke(result: dict[str, Any]) -> str:
    """Classify whether objective failures point to env, reward, or training."""
    if result["tower_damage"] <= 0.0:
        return CONCLUSION_FIX_ENV
    if result["base_exposed_rate"] <= 0.0 or result["avg_base_damage"] <= 0.0:
        return CONCLUSION_FIX_ENV
    if result["avg_reward_margin"] <= 0.0:
        return CONCLUSION_FIX_REWARD
    if result["hard_win_rate"] <= 0.0:
        return CONCLUSION_ADJUST_TRAINING
    return CONCLUSION_PASS


def run_objective_policy_smoke(
    *,
    episodes: int = 3,
    seed: int = 42,
    map_size: int = 16,
    team_size: int = 2,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Run a deterministic scripted objective-policy smoke test."""
    reward_config = RewardConfig(
        objective_enabled=True,
        objective_tower_damage_team=0.001,
        objective_base_damage_team=0.003,
        objective_base_exposed_team=1.0,
        objective_step_cap_team=0.25,
        damage=0.0,
        tower=0.0,
        tower_lost=0.0,
        base=0.0,
        win=5.0,
        lose=-5.0,
        time_penalty=0.0,
    )
    env = parallel_env(
        map_size=map_size,
        team_size=team_size,
        max_steps=max_steps,
        reward_config=reward_config,
        fog_of_war=True,
        seed=seed,
    )

    hard_wins = 0
    base_exposed = 0
    tower_damages: list[float] = []
    base_damages: list[float] = []
    red_rewards: list[float] = []
    blue_rewards: list[float] = []
    lengths: list[int] = []
    terminal_reasons: list[str | None] = []

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        assert env.game_state is not None
        attacker_id = "red_0"
        ep_red_reward = 0.0
        ep_blue_reward = 0.0
        steps = 0

        while env.agents:
            game = env.game_state
            actions = {
                agent_id: _scripted_objective_action(game, agent_id, attacker_id)
                if agent_id.startswith("red")
                else _noop_action()
                for agent_id in env.agents
            }
            _, rewards, terms, truncs, _ = env.step(actions)
            ep_red_reward += sum(
                reward for agent_id, reward in rewards.items() if agent_id.startswith("red")
            )
            ep_blue_reward += sum(
                reward for agent_id, reward in rewards.items() if agent_id.startswith("blue")
            )
            steps += 1
            if any(terms.values()) or any(truncs.values()):
                break

        game = env.game_state
        terminal_reasons.append(game.terminal_reason)
        if game.game_winner == "red" and game.terminal_reason == "base_destroyed":
            hard_wins += 1
        if game.blue_towers == 0:
            base_exposed += 1
        tower_damages.append(game.red_tower_damage)
        base_damages.append(game.red_base_damage)
        red_rewards.append(ep_red_reward)
        blue_rewards.append(ep_blue_reward)
        lengths.append(steps)

    total = max(episodes, 1)
    avg_red_reward = float(np.mean(red_rewards)) if red_rewards else 0.0
    avg_blue_reward = float(np.mean(blue_rewards)) if blue_rewards else 0.0
    result: dict[str, Any] = {
        "episodes": episodes,
        "seed": seed,
        "map_size": map_size,
        "team_size": team_size,
        "max_steps": max_steps,
        "hard_win_rate": hard_wins / total,
        "base_exposed_rate": base_exposed / total,
        "avg_base_damage": float(np.mean(base_damages)) if base_damages else 0.0,
        "tower_damage": float(np.mean(tower_damages)) if tower_damages else 0.0,
        "avg_tower_damage": float(np.mean(tower_damages)) if tower_damages else 0.0,
        "avg_red_reward": avg_red_reward,
        "avg_blue_reward": avg_blue_reward,
        "avg_reward_margin": avg_red_reward - avg_blue_reward,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "terminal_reasons": terminal_reasons,
    }
    result["conclusion"] = classify_objective_policy_smoke(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scripted objective policy smoke test")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_objective_policy_smoke(
        episodes=args.episodes,
        seed=args.seed,
        map_size=args.map_size,
        team_size=args.team_size,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
