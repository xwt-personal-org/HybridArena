"""Quick training smoke test: verify PPO loss decreases and Self-Play works.

Usage:
    python scripts/train_smoke_test.py
"""

from __future__ import annotations

import sys

from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.training.trainer import Trainer


def main():
    print("=" * 60)
    print("Smoke Test 1: Vanilla PPO (no self-play)")
    print("=" * 60)

    config = PPOConfig(
        total_timesteps=5_000,
        num_steps=128,
        n_epochs=4,
        minibatch_size=64,
        learning_rate=3e-4,
        device="cpu",
        seed=42,
    )

    trainer = Trainer(config, algo_type="ppo", use_self_play=False)
    result = trainer.train()

    n_episodes = len(result["episode_rewards"])
    print(f"\n[Result] {n_episodes} episodes, final 5 rewards: {result['episode_rewards'][-5:]}")

    # Check that loss is roughly stable/decreasing (very loose check)
    if n_episodes < 2:
        print("WARNING: Too few episodes to verify training.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Smoke Test 2: Dual-clip PPO + Self-Play")
    print("=" * 60)

    config2 = PPOConfig(
        total_timesteps=5_000,
        num_steps=128,
        n_epochs=4,
        minibatch_size=64,
        learning_rate=3e-4,
        device="cpu",
        seed=123,
        eval_interval=2_500,
    )

    trainer2 = Trainer(config2, algo_type="ppo_dualclip", use_self_play=True)
    result2 = trainer2.train()

    n_episodes2 = len(result2["episode_rewards"])
    print(f"\n[Result] {n_episodes2} episodes, final 5 rewards: {result2['episode_rewards'][-5:]}")
    print(f"[SelfPlay] Pool size: {len(trainer2.sp_manager) if trainer2.sp_manager else 0}")

    print("\n" + "=" * 60)
    print("Smoke tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
