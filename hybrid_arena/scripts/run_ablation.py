"""Run a small reproducible ablation matrix and write local reports.

Supports two modes:
    1. Baseline mode: evaluate rule-based vs random/rule-based (no training)
    2. Train+eval mode: train models first, then evaluate checkpoints
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.minimoba.reward_shaper import RewardConfig
from hybrid_arena.training.evaluator import evaluate_policy
from hybrid_arena.training.trainer import Trainer


def _build_reward_config(cfg: dict) -> RewardConfig | None:
    """Build RewardConfig from YAML reward section. Returns None if not present."""
    reward_cfg = cfg.get("reward", {})
    if not reward_cfg:
        return None
    return RewardConfig(**{k: v for k, v in reward_cfg.items() if hasattr(RewardConfig, k)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HybridArena ablation matrix")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned matrix without running"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "baseline", "train_eval", "eval_only"],
        help="Execution mode: baseline (no training), train_eval (train then eval), eval_only (eval existing checkpoints), auto (detect from config)",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=None,
        help="Override training timesteps (default: from config)",
    )
    return parser.parse_args()


def _opponent_policy(name: str):
    agent = RuleBasedAgent() if name == "rule_based" else RandomAgent()
    return agent.act


def _find_checkpoint(checkpoint_dir: Path, algo: str, seed: int) -> str | None:
    """Find the latest checkpoint for a given algorithm and seed."""
    pattern = f"{algo}_seed{seed}_step*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None
    # Sort by step number and return the latest
    checkpoints.sort(key=lambda p: int(p.stem.split("step")[-1]))
    return str(checkpoints[-1])


def _load_policy_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a policy function from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["network_state_dict"]
    config_dict = ckpt.get("config", {})

    hidden_dim = config_dict.get("hidden_dim", 48)
    network = ActorCritic(hidden_dim=hidden_dim).to(device)
    network.load_state_dict(state_dict)
    network.eval()

    def policy_fn(obs: dict, agent_id: str) -> np.ndarray:
        obs_t = {
            k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in obs.items()
            if k != "action_mask"
        }
        obs_t["action_mask"] = torch.tensor(
            obs["action_mask"], dtype=torch.int8, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action = network.get_action(obs_t)
        return action.squeeze(0).cpu().numpy()

    return policy_fn


def _train_model(algo: str, seed: int, total_timesteps: int, config: dict) -> str:
    """Train a model and return the checkpoint path."""
    checkpoint_dir = Path(config.get("outputs", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build training config
    training_cfg = config.get("training", {})
    reward_config = _build_reward_config(config)
    ppo_config = PPOConfig(
        map_size=config.get("experiment", {}).get("map_size", 32),
        team_size=config.get("experiment", {}).get("team_size", 4),
        max_steps=config.get("experiment", {}).get("max_steps", 1000),
        fog_of_war=True,
        total_timesteps=total_timesteps,
        num_envs=training_cfg.get("num_envs", 4),
        num_steps=training_cfg.get("num_steps", 128),
        device=training_cfg.get("device", "cpu"),
        seed=seed,
        reward_config=reward_config,
    )

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train
    print(f"\n{'=' * 60}")
    print(f"Training {algo} (seed={seed})")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Device: {ppo_config.device}")
    print(f"{'=' * 60}")

    trainer = Trainer(
        config=ppo_config,
        algo_type=algo,
        checkpoint_dir=str(checkpoint_dir),
        save_interval=total_timesteps,  # Save only at the end
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # Save final checkpoint
    final_step = trainer.global_timestep
    checkpoint_path = checkpoint_dir / f"{algo}_seed{seed}_step{final_step}.pt"
    print(f"[Train] Completed in {elapsed:.0f}s. Checkpoint: {checkpoint_path}")

    return str(checkpoint_path)


def _write_summary(path: Path, rows: list[dict]) -> None:
    header = (
        "| algo | seed | opponent | win_rate | draw_rate | hard_win | timeout_win | timeout_draw | "
        "avg_reward | avg_len | towers | tower_hp | tower_dmg | base_dmg | enemy_base_hp | base_exp | fps |"
    )
    sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {algo} | {seed} | {opponent} | {win_rate:.3f} | {draw_rate:.3f} | "
            "{hard_win_rate:.3f} | {timeout_win_rate:.3f} | {timeout_draw_rate:.3f} | "
            "{avg_reward:.3f} | {avg_len:.1f} | {avg_towers_destroyed:.1f} | "
            "{avg_tower_hp_advantage:.1f} | {avg_tower_damage:.1f} | {avg_base_damage:.1f} | "
            "{avg_enemy_base_hp_remaining:.1f} | {base_exposed_rate:.3f} | {fps:.1f} |".format(
                **row
            )
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
    train_timesteps = args.train_timesteps

    cfg = {}
    if args.config:
        cfg = _load_config(args.config)
        exp = cfg.get("experiment", {})
        seeds = exp.get("seeds", seeds)
        algorithms = exp.get("algorithms", algorithms)
        opponents = exp.get("opponents", opponents)
        episodes = exp.get("episodes", episodes)
        max_steps = exp.get("max_steps", max_steps)
        map_size = exp.get("map_size", map_size)
        team_size = exp.get("team_size", team_size)
        output_cfg = cfg.get("outputs", {})
        output_dir = Path(output_cfg.get("result_dir", str(output_dir)))
        if train_timesteps is None:
            train_timesteps = cfg.get("training", {}).get("total_timesteps", None)

    # Build reward config from YAML
    reward_config = _build_reward_config(cfg) if cfg else None

    # Determine mode
    mode = args.mode
    if mode == "auto":
        # If config has training section with timesteps, use train_eval mode
        mode = "train_eval" if train_timesteps and train_timesteps > 0 else "baseline"

    # Get checkpoint directory
    checkpoint_dir = (
        Path(cfg.get("outputs", {}).get("checkpoint_dir", "checkpoints"))
        if cfg
        else Path("checkpoints")
    )

    if args.dry_run:
        print(f"[dry-run] Mode: {mode}")
        print("[dry-run] Matrix:")
        for algo in algorithms:
            for seed in seeds:
                for opponent in opponents:
                    print(
                        f"  algo={algo}  seed={seed}  opponent={opponent}  "
                        f"episodes={episodes}  max_steps={max_steps}"
                    )
        print(f"[dry-run] Output dir: {output_dir}")
        print(f"[dry-run] Total runs: {len(algorithms) * len(seeds) * len(opponents)}")
        if mode == "train_eval":
            print(f"[dry-run] Training timesteps: {train_timesteps:,}")
        if mode == "eval_only":
            print(f"[dry-run] Checkpoint dir: {checkpoint_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo in algorithms:
        for seed in seeds:
            # Get policy function
            if mode == "train_eval":
                # Train model first
                checkpoint_path = _train_model(algo, seed, train_timesteps, cfg)
                policy_fn = _load_policy_from_checkpoint(checkpoint_path)
                policy_label = f"{algo}_seed{seed}"
            elif mode == "eval_only":
                # Load existing checkpoint
                checkpoint_path = _find_checkpoint(checkpoint_dir, algo, seed)
                if not checkpoint_path:
                    print(f"[Warning] No checkpoint found for {algo} seed={seed}, skipping")
                    continue
                policy_fn = _load_policy_from_checkpoint(checkpoint_path)
                policy_label = f"{algo}_seed{seed}"
            else:
                # Baseline mode: use rule-based
                policy_fn = RuleBasedAgent().act
                policy_label = "rule_based"

            # Evaluate against each opponent
            for opponent in opponents:
                print(f"\n[Eval] {policy_label} vs {opponent} (episodes={episodes})")
                eval_env_kwargs = {
                    "map_size": map_size,
                    "team_size": team_size,
                    "max_steps": max_steps,
                }
                if reward_config is not None:
                    eval_env_kwargs["reward_config"] = reward_config
                result = evaluate_policy(
                    policy_fn,
                    opponent_fn=_opponent_policy(opponent),
                    n_episodes=episodes,
                    env_kwargs=eval_env_kwargs,
                    seed_offset=seed,
                )
                rows.append(
                    {
                        "algo": algo if mode in ("train_eval", "eval_only") else "rule_based",
                        "seed": seed,
                        "opponent": opponent,
                        "win_rate": result["win_rate"],
                        "draw_rate": result["draw_rate"],
                        "hard_win_rate": result.get("hard_win_rate", 0.0),
                        "timeout_win_rate": result.get("timeout_win_rate", 0.0),
                        "timeout_draw_rate": result.get("timeout_draw_rate", 0.0),
                        "avg_reward": result["avg_reward"],
                        "avg_len": result["avg_episode_length"],
                        "avg_towers_destroyed": result["avg_towers_destroyed"],
                        "avg_tower_hp_advantage": result["avg_tower_hp_advantage"],
                        "avg_tower_damage": result.get("avg_tower_damage", 0.0),
                        "avg_base_damage": result.get("avg_base_damage", 0.0),
                        "avg_enemy_base_hp_remaining": result.get(
                            "avg_enemy_base_hp_remaining", 0.0
                        ),
                        "base_exposed_rate": result.get("base_exposed_rate", 0.0),
                        "fps": result["fps"],
                    }
                )

    raw_path = output_dir / "ablation_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "seed",
                "opponent",
                "win_rate",
                "draw_rate",
                "hard_win_rate",
                "timeout_win_rate",
                "timeout_draw_rate",
                "avg_reward",
                "avg_len",
                "avg_towers_destroyed",
                "avg_tower_hp_advantage",
                "avg_tower_damage",
                "avg_base_damage",
                "avg_enemy_base_hp_remaining",
                "base_exposed_rate",
                "fps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(output_dir / "ablation_summary.md", rows)
    print(f"\n[Done] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
