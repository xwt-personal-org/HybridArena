"""Training loop for PPO and variants on MiniMOBA.

Architecture:
    One MiniMOBAEnv (8 agents) → network batched forward over all agents
    → rollout buffer (T, N_agents) → PPO update on flat (T*N_agents, ...)

Self-play integration:
    - Training: current policy vs itself (all agents share weights).
    - Evaluation: current policy vs opponent pool every eval_interval steps.
    - Opponent pool updated via ELO gating.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.algorithms.ppo.ppo import PPO
from hybrid_arena.algorithms.ppo.ppo_dualclip import DualClipPPO
from hybrid_arena.algorithms.self_play.manager import PolicyCheckpoint, SelfPlayManager
from hybrid_arena.minimoba.env import parallel_env
from hybrid_arena.training.buffer import RolloutBuffer
from hybrid_arena.training.checkpoint import save_checkpoint
from hybrid_arena.training.evaluator import Evaluator
from hybrid_arena.training.logger import WandbLogger
from hybrid_arena.training.vector_runner import SyncParallelEnvRunner


def make_algorithm(config: PPOConfig, algo_type: str = "ppo"):
    if algo_type == "ppo_dualclip":
        return DualClipPPO(config)
    return PPO(config)


class Trainer:
    def __init__(
        self,
        config: PPOConfig,
        algo_type: str = "ppo",
        use_self_play: bool = False,
        use_curriculum: bool = False,
        wandb_config: dict | None = None,
        checkpoint_dir: str | None = None,
        log_interval: int = 1000,
        save_interval: int = 50000,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.algo_type = algo_type
        self.use_self_play = use_self_play
        self.use_curriculum = use_curriculum
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.save_interval = save_interval

        env_kwargs = {
            "map_size": config.map_size,
            "team_size": config.team_size,
            "max_steps": config.max_steps,
            "fog_of_war": config.fog_of_war,
        }
        if config.num_envs > 1:
            self.env = SyncParallelEnvRunner(
                num_envs=config.num_envs,
                env_kwargs=env_kwargs,
                seed=config.seed,
            )
        else:
            self.env = parallel_env(**env_kwargs, seed=config.seed)
        self.num_agents = len(self.env.possible_agents)
        self.agents_per_env = self.num_agents // max(config.num_envs, 1)
        self.team_size = config.team_size

        self.algorithm = make_algorithm(config, algo_type)

        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_agents=self.num_agents,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=config.device,
        )

        # Self-play
        self.sp_manager: SelfPlayManager | None = None
        if use_self_play:
            self.sp_manager = SelfPlayManager(
                pool_size=config.self_play_pool_size,
                win_threshold=0.55,
                n_eval_games=20,
            )

        # Evaluator
        self.evaluator = Evaluator(
            env_kwargs={
                "map_size": config.map_size,
                "team_size": config.team_size,
                "max_steps": config.max_steps,
                "fog_of_war": config.fog_of_war,
            },
            n_eval_episodes=20,
            eval_interval=getattr(config, "eval_interval", 30_000),
        )

        # Logger
        wandb_cfg = wandb_config or {}
        self.logger = WandbLogger(
            project=wandb_cfg.get("project", "hybrid-arena"),
            group=wandb_cfg.get("group", algo_type),
            name=wandb_cfg.get("name"),
            config=config.__dict__,
            enabled=wandb_cfg.get("enabled", False),
        )

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.global_timestep = 0
        self._win_history: list[bool] = []
        self._last_update_info: dict[str, float] = {}
        self.training_curve: list[dict] = []

    def train(self):
        config = self.config
        print(f"[Trainer] Algorithm: {self.algo_type}")
        print(f"[Trainer] Self-play: {self.use_self_play}")
        print(f"[Trainer] Timesteps: {config.total_timesteps:,}")
        print(f"[Trainer] Device: {self.device}")
        print(
            f"[Trainer] num_envs={config.num_envs} agents_per_env={self.agents_per_env} "
            f"transitions_per_rollout={config.num_steps * self.num_agents}"
        )

        obs, _ = self.env.reset(seed=config.seed)
        start_time = time.time()
        self.global_timestep = 0
        episode_reward_accum = 0.0
        episode_length = 0
        last_eval_step = 0

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

                # Track win for self-play / curriculum
                if any(terms.values()) or any(truncs.values()):
                    winner = self._get_winner()
                    if winner == "red":
                        self._win_history.append(True)
                    elif winner == "blue":
                        self._win_history.append(False)

                # Store in buffer (keyed by possible_agents order)
                obs_np = self._stack_obs_np(obs)
                action_mask_np = np.stack([obs[a]["action_mask"] for a in self.env.possible_agents])
                self.buffer.add(
                    obs_np,
                    actions_np,
                    log_prob_t.cpu().numpy(),
                    rewards_arr,
                    dones_arr,
                    value_t.cpu().numpy(),
                    action_mask_np,
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

            action_masks = batch.get("action_masks")
            epoch_infos = []
            for _ in range(config.n_epochs):
                info = self.algorithm.update(
                    batch["obs"],
                    batch["actions"],
                    batch["log_probs"],
                    batch["advantages"],
                    batch["returns"],
                    action_masks,
                    batch["old_values"],
                )
                epoch_infos.append(info)

            # Average epoch info
            avg_info = {k: np.mean([i[k] for i in epoch_infos]) for k in epoch_infos[0]}
            self._last_update_info = {k: float(v) for k, v in avg_info.items()}

            # Log
            self._log_step(start_time, avg_info)

            if self.checkpoint_dir and self.global_timestep % self.save_interval == 0:
                self._save_checkpoint()

            # Periodic evaluation + self-play pool update
            if self.global_timestep - last_eval_step >= self.evaluator.eval_interval:
                last_eval_step = self.global_timestep
                self._run_evaluation()

        elapsed = time.time() - start_time
        print(f"[Trainer] Done. {self.global_timestep:,} steps in {elapsed:.0f}s")
        if self.checkpoint_dir:
            self._save_checkpoint()
        self.logger.finish()

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "global_timestep": self.global_timestep,
            "fps": self.global_timestep / max(elapsed, 1e-6),
            "last_policy_loss": self._last_update_info.get("policy_loss", 0.0),
            "last_value_loss": self._last_update_info.get("value_loss", 0.0),
            "training_curve": self.training_curve,
        }

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_dir:
            return
        path = (
            Path(self.checkpoint_dir)
            / f"{self.algo_type}_seed{self.config.seed}_step{self.global_timestep}.pt"
        )
        save_checkpoint(
            path,
            self.algorithm.network,
            self.algorithm.optimizer,
            self.config,
            self.global_timestep,
            self._last_update_info,
        )

    def _get_winner(self) -> str | None:
        if self.env.game_state is None:
            return None
        return self.env.game_state.get_winner()

    def _log_step(self, start_time: float, info: dict) -> None:
        if not self.episode_rewards:
            return
        elapsed = time.time() - start_time
        fps = self.global_timestep / max(elapsed, 1e-6)
        n = min(10, len(self.episode_rewards))
        avg_r = np.mean(self.episode_rewards[-n:])
        avg_len = np.mean(self.episode_lengths[-n:])

        log_data = {
            "global_step": self.global_timestep,
            "reward": avg_r,
            "episode_length": avg_len,
            "fps": fps,
            "policy_loss": info.get("policy_loss", 0.0),
            "value_loss": info.get("value_loss", 0.0),
            "entropy": info.get("entropy", 0.0),
            "approx_kl": info.get("approx_kl", 0.0),
            "clip_fraction": info.get("clip_fraction", 0.0),
        }

        if "dual_clip_fraction" in info:
            log_data["dual_clip_fraction"] = info["dual_clip_fraction"]

        self.logger.log(log_data, step=self.global_timestep)
        self.training_curve.append(log_data)

        print(
            f"Step {self.global_timestep:7,d} | "
            f"Reward: {avg_r:+.2f} | "
            f"Len: {avg_len:.0f} | "
            f"FPS: {fps:.0f} | "
            f"KL: {info['approx_kl']:.4f} | "
            f"Ent: {info['entropy']:.3f} | "
            f"Clip: {info['clip_fraction']:.3f}"
        )

    def _run_evaluation(self) -> None:
        """Evaluate current policy and optionally update self-play pool."""
        print(f"\n[Eval] Running evaluation at step {self.global_timestep}")

        def policy_fn(obs, agent_id):
            # Wrap single obs into batch
            obs_b = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in obs.items() if k != "action_mask"
            }
            obs_b["action_mask"] = torch.tensor(
                obs["action_mask"], dtype=torch.int8, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self.algorithm.get_action(obs_b)
            return action.squeeze(0).cpu().numpy()

        # Evaluate vs rule-based baseline
        from hybrid_arena.minimoba.agents.rule_based import RuleBasedAgent
        rule_agent = RuleBasedAgent()

        def opponent_fn(obs, agent_id):
            return rule_agent.act(obs)

        result = self.evaluator.evaluate(
            policy_fn, opponent_fn=opponent_fn, global_step=self.global_timestep
        )
        print(
            f"[Eval] Win rate vs rule-based: {result['win_rate']:.1%} "
            f"(avg_reward={result['avg_reward']:+.2f})"
        )

        # Update self-play pool if enabled
        if self.use_self_play and self.sp_manager is not None:
            self._maybe_update_pool(policy_fn, result["win_rate"])

    def _maybe_update_pool(self, policy_fn, win_rate: float) -> None:
        """Add current policy to opponent pool if it passes quality gate."""
        current_state = copy.deepcopy(self.algorithm.network.state_dict())

        def eval_fn(current: PolicyCheckpoint, opponent: PolicyCheckpoint) -> float:
            # Quick eval: load opponent weights and play a few games
            opp_net = copy.deepcopy(self.algorithm.network)
            opp_net.load_state_dict(opponent.state_dict)
            opp_net.eval()

            wins, games = 0, 0
            for _ in range(5):
                obs, _ = self.env.reset()
                while self.env.agents:
                    obs_t = self._stack_obs(obs)
                    with torch.no_grad():
                        action_t, _, _ = self.algorithm.get_action(obs_t)
                    actions_np = action_t.cpu().numpy()
                    action_dict = {}
                    for i, aid in enumerate(self.env.possible_agents):
                        if aid in self.env.agents:
                            action_dict[aid] = actions_np[i]
                    obs, _, terms, truncs, _ = self.env.step(action_dict)
                    if any(terms.values()) or any(truncs.values()):
                        break
                if self.env.game_state and self.env.game_state.get_winner() == "red":
                    wins += 1
                games += 1
            return wins / games if games > 0 else 0.0

        added = self.sp_manager.maybe_add_to_pool(current_state, eval_fn)
        status = "ADDED" if added else "REJECTED"
        print(f"[SelfPlay] Pool size={len(self.sp_manager)} | Checkpoint {status}")
        if self.sp_manager.policy_pool:
            top = self.sp_manager.get_elo_leaderboard(3)
            print(f"[SelfPlay] Top ELO: {top}")

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
