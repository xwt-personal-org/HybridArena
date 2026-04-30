"""Evaluation script for trained policies.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/ppo_seed42_step10000.pt \
        --opponent rule_based --n-episodes 100

    python scripts/evaluate.py --checkpoint checkpoints/ppo_seed42_step10000.pt \
        --opponent self --n-episodes 100

    python scripts/evaluate.py --baseline rule_based --n-episodes 200
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
from hybrid_arena.training.evaluator import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained policies")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (omit for --baseline)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "ppo_dualclip", "mappo", "qmix", "coma"],
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="rule_based",
        choices=["rule_based", "self", "random"],
        help="Opponent type",
    )
    parser.add_argument("--episodes", "--n-episodes", dest="n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)

    # Baseline mode: evaluate rule-based vs itself
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["rule_based", "random"],
        help="Run baseline evaluation instead of loading a checkpoint",
    )

    return parser.parse_args()


def load_policy_from_checkpoint(
    checkpoint_path: str,
    algorithm: str,
    device: str = "cpu",
) -> tuple[Callable, dict]:
    """Load a policy function from a checkpoint file.

    Returns:
        policy_fn(obs_dict, agent_id) -> action array
        metadata dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["network_state_dict"]
    config_dict = ckpt.get("config", {})

    hidden_dim = config_dict.get("hidden_dim", 48)
    network = ActorCritic(hidden_dim=hidden_dim).to(device)
    network.load_state_dict(state_dict)
    network.eval()

    def policy_fn(obs: dict, agent_id: str) -> np.ndarray:
        # Convert obs dict to tensors
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

    return policy_fn, ckpt


def get_opponent_fn(opponent_type: str, device: str = "cpu") -> Callable | None:
    """Get opponent policy function."""
    if opponent_type == "rule_based":
        agent = RuleBasedAgent()
        return lambda obs, aid: agent.act(obs)
    elif opponent_type == "random":
        agent = RandomAgent()
        return lambda obs, aid: agent.act(obs)
    elif opponent_type == "self":
        return None  # self-play: policy vs itself
    return None


def print_results(result: dict, label: str = "Evaluation") -> None:
    print("=" * 50)
    print(f"{label} Results")
    print("=" * 50)
    print(f"  Win Rate:     {result['win_rate']:.1%}")
    print(f"  Red Wins:     {result['red_wins']}")
    print(f"  Blue Wins:    {result['blue_wins']}")
    print(f"  Draws:        {result['draws']}")
    print(f"  Avg Reward:   {result['avg_reward']:+.3f}")
    print(f"  Avg Length:   {result['avg_length']:.1f}")
    if "total_kills_red" in result:
        print(f"  Total Kills:  Red={result['total_kills_red']} Blue={result['total_kills_blue']}")
    print("=" * 50)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env_kwargs = {
        "map_size": 32,
        "team_size": 4,
        "max_steps": 1000,
        "fog_of_war": True,
    }

    if args.baseline:
        # Baseline evaluation mode
        print(f"[Baseline] Running {args.baseline} vs itself...")
        opponent_fn = get_opponent_fn(args.baseline)
        result = evaluate_policy(
            policy_fn=opponent_fn,
            opponent_fn=opponent_fn,
            n_episodes=args.n_episodes,
            env_kwargs=env_kwargs,
            seed_offset=args.seed,
        )
        print_results(result, label=f"Baseline: {args.baseline}")
    else:
        if args.checkpoint and not Path(args.checkpoint).exists():
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            return

        if args.checkpoint:
            print(f"[Eval] Loading checkpoint: {args.checkpoint}")
            policy_fn, metadata = load_policy_from_checkpoint(
                args.checkpoint, args.algorithm, args.device
            )

            print(f"[Eval] Algorithm: {metadata.get('algorithm', args.algorithm)}")
            print(f"[Eval] Global step: {metadata.get('global_step', 'unknown')}")
        else:
            print("[Eval] No checkpoint supplied; using rule_based policy as baseline.")
            policy_agent = RuleBasedAgent()

            def policy_fn(obs, aid):
                return policy_agent.act(obs)

        opponent_fn = get_opponent_fn(args.opponent)
        label = f"Policy vs {args.opponent}"

        result = evaluate_policy(
            policy_fn=policy_fn,
            opponent_fn=opponent_fn,
            n_episodes=args.n_episodes,
            env_kwargs=env_kwargs,
            seed_offset=args.seed,
        )
        print_results(result, label=label)

    # Save results
    out_path = args.output
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({**result, "args": vars(args)}, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()
