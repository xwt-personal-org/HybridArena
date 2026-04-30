"""Unified training CLI for HybridArena.

Supports:
    python scripts/train.py --algorithm ppo --seed 42 --total-timesteps 3000000
    python scripts/train.py --algorithm ppo_dualclip --use-self-play --seed 42
    python scripts/train.py --algorithm mappo --seed 42
    python scripts/train.py --algorithm qmix --total-timesteps 3000000
    python scripts/train.py --algorithm coma --total-timesteps 3000000
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DRL agents on MiniMOBA")

    # Algorithm
    parser.add_argument(
        "--algo",
        "--algorithm",
        dest="algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "ppo_dualclip"],
        help="DRL algorithm to train",
    )

    # Training budget
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=4)

    # PPO core
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--dual-clip-c", type=float, default=3.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Loss coefficients
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--normalize-advantage", action="store_true", default=True)

    # Entropy schedule
    parser.add_argument(
        "--entropy-schedule",
        type=str,
        default="linear_decay",
        choices=["linear_decay", "cosine_decay", "fixed"],
    )
    parser.add_argument("--entropy-start", type=float, default=0.05)
    parser.add_argument("--entropy-end", type=float, default=0.001)

    # Network
    parser.add_argument("--hidden-dim", type=int, default=48)

    # Environment
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--team-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fog-of-war", action="store_true", default=True)

    # Self-play / curriculum
    parser.add_argument("--use-self-play", action="store_true")
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--self-play-pool-size", type=int, default=10)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=30_000)
    parser.add_argument("--n-eval-episodes", type=int, default=50)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="hybrid-arena")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-enabled", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-dir", "--save-dir", dest="save_dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=100_000)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PPOConfig:
    """Build PPOConfig from CLI args."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return PPOConfig(
        # Environment
        map_size=args.map_size,
        team_size=args.team_size,
        max_steps=args.max_steps,
        fog_of_war=args.fog_of_war,
        # Training
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        # PPO
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        dual_clip_c=args.dual_clip_c,
        max_grad_norm=args.max_grad_norm,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        normalize_advantage=args.normalize_advantage,
        entropy_schedule=args.entropy_schedule,
        entropy_start=args.entropy_start,
        entropy_end=args.entropy_end,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        use_self_play=args.use_self_play,
        self_play_pool_size=args.self_play_pool_size,
        eval_interval=args.eval_interval,
        device=device,
    )


def save_checkpoint(trainer: Trainer, save_dir: str, step: int) -> None:
    """Save policy checkpoint to disk."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(save_dir) / f"{trainer.algo_type}_seed{trainer.config.seed}_step{step}.pt"

    # Save network state dict
    state = {
        "algorithm": trainer.algo_type,
        "global_step": trainer.global_timestep,
        "network_state_dict": trainer.algorithm.network.state_dict(),
        "optimizer_state_dict": trainer.algorithm.optimizer.state_dict(),
        "config": trainer.config.__dict__,
    }

    # For MAPPO, also save critic
    if trainer.algo_type == "mappo":
        state["critic_state_dict"] = trainer.algorithm.critic.state_dict()

    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved to {ckpt_path}")


def main():
    args = parse_args()
    config = build_config(args)

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    wandb_cfg = {
        "project": args.wandb_project,
        "group": args.wandb_group or args.algorithm,
        "name": args.wandb_name or f"{args.algorithm}_seed{args.seed}",
        "enabled": args.wandb_enabled,
    }

    print("=" * 60)
    print("HybridArena Training")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Seed: {args.seed}")
    print(f"  Timesteps: {args.total_timesteps:,}")
    print(f"  Device: {config.device}")
    print(f"  Self-play: {args.use_self_play}")
    print(f"  Curriculum: {args.use_curriculum}")
    print("=" * 60)

    trainer = Trainer(
        config=config,
        algo_type=args.algorithm,
        use_self_play=args.use_self_play,
        use_curriculum=args.use_curriculum,
        wandb_config=wandb_cfg,
        checkpoint_dir=args.save_dir,
        save_interval=args.save_interval,
    )

    # Train
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    # Final save
    final_step = trainer.global_timestep
    save_checkpoint(trainer, args.save_dir, final_step)

    # Save results
    result_path = Path(args.save_dir) / f"{args.algorithm}_seed{args.seed}_results.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(
            {
                "episode_rewards": result["episode_rewards"],
                "episode_lengths": result["episode_lengths"],
                "config": config.__dict__,
                "elapsed_seconds": elapsed,
            },
            f,
        )

    print(f"\n[Done] Training finished in {elapsed:.0f}s")
    print(f"[Done] Final checkpoint saved to {args.save_dir}")
    print(f"[Done] Results saved to {result_path}")


if __name__ == "__main__":
    main()
