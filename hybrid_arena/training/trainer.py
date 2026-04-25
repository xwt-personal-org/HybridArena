"""Training loop for PPO and variants on MiniMOBA.

Architecture:
    One MiniMOBAEnv (8 agents) → network batched forward over all agents
    → rollout buffer (T, N_agents) → PPO update on flat (T*N_agents, ...)
"""

from __future__ import annotations

import time

import numpy as np
import torch

from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.algorithms.ppo.ppo import PPO
from hybrid_arena.algorithms.ppo.ppo_dualclip import DualClipPPO
from hybrid_arena.minimoba.env import parallel_env
from hybrid_arena.training.buffer import RolloutBuffer


def make_algorithm(config: PPOConfig, algo_type: str = "ppo"):
    if algo_type == "ppo_dualclip":
        return DualClipPPO(config)
    return PPO(config)


class Trainer:
    def __init__(
        self,
        config: PPOConfig,
        algo_type: str = "ppo",
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.algo_type = algo_type

        self.env = parallel_env(
            map_size=config.map_size,
            team_size=config.team_size,
            max_steps=config.max_steps,
            fog_of_war=config.fog_of_war,
            seed=config.seed,
        )
        self.num_agents = len(self.env.possible_agents)  # 8 for 4v4

        self.algorithm = make_algorithm(config, algo_type)

        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_agents=self.num_agents,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=config.device,
        )

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.global_timestep = 0

    def train(self):
        config = self.config
        print(f"[Trainer] Algorithm: {self.algo_type}")
        print(f"[Trainer] Timesteps: {config.total_timesteps:,}")
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] {config.num_steps} steps/rollout × {self.num_agents} agents")

        obs, _ = self.env.reset(seed=config.seed)
        start_time = time.time()
        self.global_timestep = 0
        episode_reward_accum = 0.0
        episode_length = 0

        while self.global_timestep < config.total_timesteps:
            self.buffer.reset()

            for _ in range(config.num_steps):
                if not self.env.agents:
                    obs, _ = self.env.reset()

                # Batch all agents through network
                obs_tensors = self._stack_obs(obs)
                with torch.no_grad():
                    action_t, log_prob_t, value_t = self.algorithm.get_action(
                        obs_tensors, action_mask=obs_tensors.get("action_mask")
                    )

                # Build action dict for env step
                actions_np = action_t.cpu().numpy()  # (N, 3)
                action_dict = {}
                for i, aid in enumerate(self.env.possible_agents):
                    if aid in self.env.agents:
                        action_dict[aid] = actions_np[i]

                # Step
                next_obs, rewards_dict, terms, truncs, _ = self.env.step(action_dict)
                _ = rewards_dict  # reserved

                # Per-agent rewards and dones
                rewards_arr = np.array(
                    [rewards_dict.get(a, 0.0) for a in self.env.possible_agents],
                    dtype=np.float32,
                )
                dones_arr = np.array(
                    [
                        float(terms.get(a, False) or truncs.get(a, False))
                        for a in self.env.possible_agents
                    ],
                    dtype=np.float32,
                )

                # Store in buffer (keyed by possible_agents order)
                obs_np = self._stack_obs_np(obs)
                self.buffer.add(
                    obs_np,
                    actions_np,
                    log_prob_t.cpu().numpy(),
                    rewards_arr,
                    dones_arr,
                    value_t.cpu().numpy(),
                )

                episode_reward_accum += rewards_arr.mean()
                episode_length += 1
                obs = next_obs

                if any(terms.values()) or any(truncs.values()):
                    self.episode_rewards.append(episode_reward_accum)
                    self.episode_lengths.append(episode_length)
                    episode_reward_accum = 0.0
                    episode_length = 0

                self.global_timestep += 1
                if self.global_timestep >= config.total_timesteps:
                    break

            # Final values for GAE tail
            if self.env.agents:
                final_tensors = self._stack_obs(obs)
                with torch.no_grad():
                    _, _, next_values_t = self.algorithm.get_action(
                        final_tensors, action_mask=final_tensors.get("action_mask")
                    )
                next_values = next_values_t.cpu().numpy()
                next_dones = np.zeros(self.num_agents, dtype=np.float32)
            else:
                next_values = np.zeros(self.num_agents, dtype=np.float32)
                next_dones = np.ones(self.num_agents, dtype=np.float32)

            # Get batch and update
            batch = self.buffer.get_batch(next_values, next_dones)

            if config.normalize_advantage:
                adv = batch["advantages"]
                batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(config.n_epochs):
                info = self.algorithm.update(
                    batch["obs"],
                    batch["actions"],
                    batch["log_probs"],
                    batch["advantages"],
                    batch["returns"],
                    None,  # action_masks not stored in buffer yet
                )

            # Log
            if self.episode_rewards:
                elapsed = time.time() - start_time
                fps = self.global_timestep / max(elapsed, 1e-6)
                n = min(10, len(self.episode_rewards))
                avg_r = np.mean(self.episode_rewards[-n:])
                avg_len = np.mean(self.episode_lengths[-n:])
                print(
                    f"Step {self.global_timestep:7,d} | "
                    f"Reward: {avg_r:+.2f} | "
                    f"Len: {avg_len:.0f} | "
                    f"FPS: {fps:.0f} | "
                    f"KL: {info['approx_kl']:.4f} | "
                    f"Ent: {info['entropy']:.3f} | "
                    f"Clip: {info['clip_fraction']:.3f}"
                )

        elapsed = time.time() - start_time
        print(f"[Trainer] Done. {self.global_timestep:,} steps in {elapsed:.0f}s")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }

    def _stack_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        """Stack per-agent obs into batched tensors on device."""
        agents = self.env.possible_agents
        return {
            "local_map": torch.tensor(
                np.stack([obs[a]["local_map"] for a in agents]),
                dtype=torch.float32, device=self.device,
            ),
            "self_state": torch.tensor(
                np.stack([obs[a]["self_state"] for a in agents]),
                dtype=torch.float32, device=self.device,
            ),
            "teammate_states": torch.tensor(
                np.stack([obs[a]["teammate_states"] for a in agents]),
                dtype=torch.float32, device=self.device,
            ),
            "global_info": torch.tensor(
                np.stack([obs[a]["global_info"] for a in agents]),
                dtype=torch.float32, device=self.device,
            ),
            "action_mask": torch.tensor(
                np.stack([obs[a]["action_mask"] for a in agents]),
                dtype=torch.int8, device=self.device,
            ),
        }

    def _stack_obs_np(self, obs: dict) -> dict[str, np.ndarray]:
        """Stack per-agent obs into numpy arrays for buffer storage."""
        agents = self.env.possible_agents
        return {
            "local_map": np.stack([obs[a]["local_map"] for a in agents]),
            "self_state": np.stack([obs[a]["self_state"] for a in agents]),
            "teammate_states": np.stack([obs[a]["teammate_states"] for a in agents]),
            "global_info": np.stack([obs[a]["global_info"] for a in agents]),
        }
