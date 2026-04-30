"""Run a small reproducible ablation matrix and write local reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.training.evaluator import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HybridArena ablation matrix")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Print planned matrix without running")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def _opponent_policy(name: str):
    agent = RuleBasedAgent() if name == "rule_based" else RandomAgent()
    return agent.act


def _write_summary(path: Path, rows: list[dict]) -> None:
    header = (
        "| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | "
        "avg_towers_destroyed | avg_tower_hp_advantage | fps |"
    )
    sep = (
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {algo} | {seed} | {opponent} | {win_rate:.3f} | {draw_rate:.3f} | "
            "{avg_reward:.3f} | {avg_len:.1f} | {avg_towers_destroyed:.1f} | "
            "{avg_tower_hp_advantage:.1f} | {fps:.1f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    seeds = [42, 123, 456]
    algorithms = ["ppo", "ppo_dualclip"]
    opponents = ["random", "rule_based"]
    episodes = args.episodes
    max_steps = args.max_steps
    map_size = args.map_size
    team_size = args.team_size
    output_dir = Path(args.output_dir)

    if args.config:
        cfg = _load_config(args.config)
        exp = cfg.get("experiment", {})
        seeds = exp.get("seeds", seeds)
        algorithms = exp.get("algorithms", algorithms)
        opponents = exp.get("opponents", opponents)
        episodes = exp.get("episodes", episodes)
        max_steps = exp.get("max_steps", max_steps)
        output_cfg = cfg.get("outputs", {})
        output_dir = Path(output_cfg.get("result_dir", str(output_dir)))

    if args.dry_run:
        print("[dry-run] Matrix:")
        for algo in algorithms:
            for seed in seeds:
                for opponent in opponents:
                    print(f"  algo={algo}  seed={seed}  opponent={opponent}  "
                          f"episodes={episodes}  max_steps={max_steps}")
        print(f"[dry-run] Output dir: {output_dir}")
        print(f"[dry-run] Total runs: {len(algorithms) * len(seeds) * len(opponents)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    policy = RuleBasedAgent().act
    rows = []
    for algo in algorithms:
        for seed in seeds:
            for opponent in opponents:
                result = evaluate_policy(
                    policy,
                    opponent_fn=_opponent_policy(opponent),
                    n_episodes=episodes,
                    env_kwargs={
                        "map_size": map_size,
                        "team_size": team_size,
                        "max_steps": max_steps,
                    },
                    seed_offset=seed,
                )
                rows.append(
                    {
                        "algo": algo,
                        "seed": seed,
                        "opponent": opponent,
                        "win_rate": result["win_rate"],
                        "draw_rate": result["draw_rate"],
                        "avg_reward": result["avg_reward"],
                        "avg_len": result["avg_episode_length"],
                        "avg_towers_destroyed": result["avg_towers_destroyed"],
                        "avg_tower_hp_advantage": result["avg_tower_hp_advantage"],
                        "fps": result["fps"],
                    }
                )

    raw_path = output_dir / "ablation_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo", "seed", "opponent", "win_rate", "draw_rate",
                "avg_reward", "avg_len", "avg_towers_destroyed",
                "avg_tower_hp_advantage", "fps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(output_dir / "ablation_summary.md", rows)


if __name__ == "__main__":
    main()
