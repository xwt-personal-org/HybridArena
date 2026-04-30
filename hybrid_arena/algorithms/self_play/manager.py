"""Self-play opponent pool manager with ELO-based quality gate.

Design philosophy (interview key point):
    1. Maintain a pool of historical checkpoints; sample opponents from it.
    2. New policy must beat recent pool policies (>win_threshold) to enter.
    3. This prevents overfitting to the current opponent and ensures
       monotonically improving policy quality.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from hybrid_arena.algorithms.self_play.elo import ELORatingSystem


class PolicyCheckpoint:
    """Lightweight wrapper around a policy state dict + metadata."""

    def __init__(self, policy_id: str, state_dict: dict, metadata: dict | None = None):
        self.policy_id = policy_id
        self.state_dict = state_dict
        self.metadata = metadata or {}

    def clone(self) -> PolicyCheckpoint:
        return PolicyCheckpoint(
            policy_id=self.policy_id,
            state_dict=copy.deepcopy(self.state_dict),
            metadata=copy.deepcopy(self.metadata),
        )


class SelfPlayManager:
    """Manages a pool of historical policies for self-play training.

    Args:
        pool_size: Max number of checkpoints to keep.
        win_threshold: Minimum win rate vs recent pool to qualify for entry.
        n_eval_games: Number of evaluation games per qualification check.
        recent_window: Number of most recent policies to evaluate against.
        sample_recent_prob: Probability of sampling from recent policies.
    """

    def __init__(
        self,
        pool_size: int = 10,
        win_threshold: float = 0.55,
        n_eval_games: int = 50,
        recent_window: int = 3,
        sample_recent_prob: float = 0.8,
    ):
        self.pool_size = pool_size
        self.win_threshold = win_threshold
        self.n_eval_games = n_eval_games
        self.recent_window = recent_window
        self.sample_recent_prob = sample_recent_prob

        self.policy_pool: list[PolicyCheckpoint] = []
        self.elo = ELORatingSystem()
        self._step_counter = 0

    def get_opponent(self) -> PolicyCheckpoint | None:
        """Sample an opponent from the pool.

        Strategy: 80% chance from recent 5, 20% uniform from entire pool.
        Falls back to None if pool is empty (caller uses rule-based baseline).
        """
        if not self.policy_pool:
            return None

        import numpy as np

        if np.random.random() < self.sample_recent_prob and len(self.policy_pool) >= 5:
            idx = int(np.random.choice(min(5, len(self.policy_pool))))
            return self.policy_pool[-(idx + 1)]
        else:
            idx = int(np.random.randint(0, len(self.policy_pool)))
            return self.policy_pool[idx]

    def maybe_add_to_pool(
        self,
        current_policy_state: dict,
        eval_fn: Callable[[PolicyCheckpoint, PolicyCheckpoint], float],
        policy_id: str | None = None,
    ) -> bool:
        """Evaluate current policy and add to pool if it passes the gate.

        Args:
            current_policy_state: State dict of the current trained policy.
            eval_fn: Function (policy_a, policy_b) -> win_rate of A vs B.
            policy_id: Optional ID; auto-generated if None.

        Returns:
            True if added to pool.
        """
        import numpy as np

        pid = policy_id or f"policy_{self._step_counter}"
        self._step_counter += 1

        current_ckpt = PolicyCheckpoint(pid, copy.deepcopy(current_policy_state))

        if not self.policy_pool:
            self.policy_pool.append(current_ckpt)
            self.elo.register(pid)
            return True

        # Evaluate against the most recent `recent_window` opponents
        opponents = self.policy_pool[-self.recent_window :]
        win_rates = []

        for opp in opponents:
            wr = eval_fn(current_ckpt, opp)
            win_rates.append(wr)
            # Update ELO after each pairwise eval
            self.elo.update(pid, opp.policy_id, winner=pid if wr > 0.5 else (opp.policy_id if wr < 0.5 else None))

        avg_win_rate = float(np.mean(win_rates))

        if avg_win_rate >= self.win_threshold:
            self.policy_pool.append(current_ckpt)
            # Maintain pool size
            if len(self.policy_pool) > self.pool_size:
                self.policy_pool.pop(0)
                # Note: we keep ELO records for history; don't delete
            return True

        return False

    def get_elo_leaderboard(self, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k policies by ELO rating."""
        return self.elo.top_k(k)

    def __len__(self) -> int:
        return len(self.policy_pool)

    def state_dict(self) -> dict:
        """Serialize manager state."""
        return {
            "pool_size": self.pool_size,
            "win_threshold": self.win_threshold,
            "n_eval_games": self.n_eval_games,
            "recent_window": self.recent_window,
            "sample_recent_prob": self.sample_recent_prob,
            "step_counter": self._step_counter,
            "pool_ids": [ckpt.policy_id for ckpt in self.policy_pool],
            "elo_records": {
                pid: {
                    "rating": rec.rating,
                    "games": rec.games,
                    "wins": rec.wins,
                    "draws": rec.draws,
                    "losses": rec.losses,
                }
                for pid, rec in self.elo._records.items()
            },
        }
