"""Observation helpers for pheromone layer integration.

These are optional helpers only — they do NOT modify the default MiniMOBA
observation contract or env.py observation generation.
"""

from __future__ import annotations

import numpy as np

from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace


def crop_pheromone_layer(
    workspace_layer: np.ndarray,
    center: tuple[int, int],
    view_size: int = 11,
) -> np.ndarray:
    """Crop a centered view from the workspace observation layer.

    Handles boundary conditions by zero-padding areas outside the map.

    Args:
        workspace_layer: Full map observation layer, shape (H, W, C).
        center: Center position (x, y) for the crop.
        view_size: Side length of the square crop (default 11).

    Returns:
        Cropped array of shape (view_size, view_size, C).
    """
    h, w = workspace_layer.shape[0], workspace_layer.shape[1]
    num_channels = workspace_layer.shape[2] if workspace_layer.ndim == 3 else 1
    cx, cy = center

    # Compute crop bounds
    half = view_size // 2
    y_min = cy - half
    y_max = cy - half + view_size
    x_min = cx - half
    x_max = cx - half + view_size

    # Initialize output with zeros (handles boundary padding)
    crop = np.zeros((view_size, view_size, num_channels), dtype=workspace_layer.dtype)

    # Compute overlap region
    src_y_min = max(y_min, 0)
    src_y_max = min(y_max, h)
    src_x_min = max(x_min, 0)
    src_x_max = min(x_max, w)

    dst_y_min = src_y_min - y_min
    dst_y_max = dst_y_min + (src_y_max - src_y_min)
    dst_x_min = src_x_min - x_min
    dst_x_max = dst_x_min + (src_x_max - src_x_min)

    if src_y_max > src_y_min and src_x_max > src_x_min:
        crop[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = (
            workspace_layer[src_y_min:src_y_max, src_x_min:src_x_max]
        )

    return crop


def append_pheromone_channels(
    local_map: np.ndarray,
    pheromone_crop: np.ndarray,
) -> np.ndarray:
    """Append pheromone channels to the local map observation.

    Concatenates along the channel dimension.

    Args:
        local_map: Local map observation, shape (11, 11, 11).
        pheromone_crop: Pheromone layer crop, shape (11, 11, C).

    Returns:
        Combined array of shape (11, 11, 11 + C).

    Raises:
        ValueError: If spatial dimensions don't match.
    """
    if local_map.shape[0] != pheromone_crop.shape[0] or local_map.shape[1] != pheromone_crop.shape[1]:
        raise ValueError(
            f"Spatial dimensions must match: local_map {local_map.shape[:2]} "
            f"vs pheromone_crop {pheromone_crop.shape[:2]}"
        )
    return np.concatenate([local_map, pheromone_crop], axis=2)


def build_augmented_observation(
    observation: dict,
    workspace: BattlefieldWorkspace,
    agent_position: tuple[int, int],
    view_size: int = 11,
) -> dict:
    """Return an opt-in observation copy with appended pheromone channels.

    The original ``local_map`` entry is kept unchanged. The new
    ``local_map_with_pheromones`` entry appends the workspace's three annotation
    channels to the original local map.
    """
    local_map = observation["local_map"]
    workspace_layer = workspace.to_observation_layer(num_channels=3)
    pheromone_crop = crop_pheromone_layer(
        workspace_layer=workspace_layer,
        center=agent_position,
        view_size=view_size,
    )
    augmented = dict(observation)
    augmented["local_map_with_pheromones"] = append_pheromone_channels(
        local_map,
        pheromone_crop,
    )
    return augmented
