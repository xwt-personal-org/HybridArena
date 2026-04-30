"""Run a small reproducible ablation matrix and write local reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.training.evaluator import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HybridArena ablation matrix")
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
    lines = [
        "| algo | seed | opponent | win_rate | avg_reward | avg_len | fps |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {algo} | {seed} | {opponent} | {win_rate:.3f} | {avg_reward:.3f} | "
            "{avg_len:.1f} | {fps:.1f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = RuleBasedAgent().act
    rows = []
    for algo in ("ppo", "ppo_dualclip"):
        for seed in (42, 123, 456):
            for opponent in ("random", "rule_based"):
                result = evaluate_policy(
                    policy,
                    opponent_fn=_opponent_policy(opponent),
                    n_episodes=args.episodes,
                    env_kwargs={
                        "map_size": args.map_size,
                        "team_size": args.team_size,
                        "max_steps": args.max_steps,
                    },
                    seed_offset=seed,
                )
                rows.append(
                    {
                        "algo": algo,
                        "seed": seed,
                        "opponent": opponent,
                        "win_rate": result["win_rate"],
                        "avg_reward": result["avg_reward"],
                        "avg_len": result["avg_episode_length"],
                        "fps": result["fps"],
                    }
                )

    raw_path = output_dir / "ablation_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algo", "seed", "opponent", "win_rate", "avg_reward", "avg_len", "fps"],
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(output_dir / "ablation_summary.md", rows)


if __name__ == "__main__":
    main()
