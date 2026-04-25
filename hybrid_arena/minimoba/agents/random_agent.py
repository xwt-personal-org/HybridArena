"""Random agent — samples from the action space uniformly."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Agent that samples random valid actions."""

    def act(self, obs: dict) -> np.ndarray:
        """Sample a random action respecting the action mask.

        Args:
            obs: Observation dict with an 'action_mask' key.

        Returns:
            np.ndarray of shape (3,) with [move_dir, skill_choice, target_choice].
        """
        mask = obs.get("action_mask")
        if mask is None:
            # No mask — sample uniformly
            return np.array(
                [
                    np.random.randint(0, 9),
                    np.random.randint(0, 4),
                    np.random.randint(0, 9),
                ],
                dtype=np.int64,
            )

        # Sample a valid action from the mask
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return np.array([0, 3, 8], dtype=np.int64)

        chosen = np.random.choice(valid_indices)
        move = chosen // 36
        skill = (chosen % 36) // 9
        target = chosen % 9
        return np.array([move, skill, target], dtype=np.int64)
