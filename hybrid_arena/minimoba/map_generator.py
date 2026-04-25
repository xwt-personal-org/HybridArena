"""Map generation for MiniMOBA: terrain, obstacles, bases, towers, jungle camps."""

from __future__ import annotations

import numpy as np

# Terrain codes
EMPTY = 0
OBSTACLE = 1
BUSH = 2
RED_BASE = 3
BLUE_BASE = 4
RED_TOWER = 5
BLUE_TOWER = 6

TERRAIN_NAMES = {
    EMPTY: "path",
    OBSTACLE: "obstacle",
    BUSH: "bush",
    RED_BASE: "red_base",
    BLUE_BASE: "blue_base",
    RED_TOWER: "red_tower",
    BLUE_TOWER: "blue_tower",
}

WALKABLE = {EMPTY, BUSH}


def generate_map(
    map_size: int = 32,
    team_size: int = 4,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a symmetric MOBA map.

    Returns:
        terrain: (map_size, map_size) int8 grid.
        spawn_points: dict with "red": [(x,y), ...], "blue": [(x,y), ...].
    """
    rng = np.random.RandomState(seed)
    terrain = np.zeros((map_size, map_size), dtype=np.int8)

    half = map_size // 2
    margin = 3

    # Red base (bottom-left area)
    terrain[-margin:, :margin] = RED_BASE
    red_spawn_center = (map_size - margin - 1, margin)

    # Blue base (top-right area)
    terrain[:margin, -margin:] = BLUE_BASE
    blue_spawn_center = (margin, map_size - margin - 1)

    # Red outer towers
    _place_tower_area(terrain, map_size - margin, map_size // 2 - 1, RED_TOWER, rng)
    _place_tower_area(terrain, map_size // 2 - 1, margin, RED_TOWER, rng)

    # Blue outer towers
    _place_tower_area(terrain, margin, map_size // 2 + 1, BLUE_TOWER, rng)
    _place_tower_area(terrain, map_size // 2 + 1, map_size - margin, BLUE_TOWER, rng)

    # Middle lane bush clusters
    for _ in range(6):
        bx = rng.randint(half - 4, half + 4)
        by = rng.randint(half - 4, half + 4)
        if terrain[by, bx] == EMPTY:
            _place_cluster(terrain, bx, by, BUSH, size=2, rng=rng)

    # Scattered bushes near corners
    for corner_x, corner_y in [(margin + 2, margin + 2), (map_size - margin - 2, map_size - margin - 2)]:
        for _ in range(2):
            bx = corner_x + rng.randint(-3, 3)
            by = corner_y + rng.randint(-3, 3)
            if 0 <= bx < map_size and 0 <= by < map_size and terrain[by, bx] == EMPTY:
                _place_cluster(terrain, bx, by, BUSH, size=1, rng=rng)

    # Sparse obstacles (fewer to prevent path blocking)
    for _ in range(map_size // 3):
        ox = rng.randint(0, map_size)
        oy = rng.randint(0, map_size)
        if terrain[oy, ox] == EMPTY:
            terrain[oy, ox] = OBSTACLE

    # Ensure no obstacle fully blocks a base exit
    _clear_path_circle(terrain, *red_spawn_center, radius=3)
    _clear_path_circle(terrain, *blue_spawn_center, radius=3)

    # Spawn points distributed in a small circle around base
    spawn_points = {
        "red": _distribute_spawns(red_spawn_center[1], red_spawn_center[0], team_size, rng),
        "blue": _distribute_spawns(blue_spawn_center[1], blue_spawn_center[0], team_size, rng),
    }

    return terrain, spawn_points


def _place_tower_area(
    terrain: np.ndarray, y: int, x: int, tower_code: int, rng: np.random.RandomState
):
    """Place a tower on the map."""
    y = max(0, min(terrain.shape[0] - 1, y))
    x = max(0, min(terrain.shape[1] - 1, x))
    terrain[y, x] = tower_code


def _place_cluster(
    terrain: np.ndarray, cx: int, cy: int, tile: int, size: int, rng: np.random.RandomState
):
    for _ in range(size * 2):
        dx = rng.randint(-size, size + 1)
        dy = rng.randint(-size, size + 1)
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < terrain.shape[1] and 0 <= ny < terrain.shape[0]:
            if terrain[ny, nx] == EMPTY:
                terrain[ny, nx] = tile


def _clear_path_circle(terrain: np.ndarray, y: int, x: int, radius: int):
    h, w = terrain.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if terrain[ny, nx] not in (RED_BASE, BLUE_BASE, RED_TOWER, BLUE_TOWER):
                        terrain[ny, nx] = EMPTY


def _distribute_spawns(
    x: int, y: int, count: int, rng: np.random.RandomState
) -> list[tuple[int, int]]:
    """Distribute spawn points in a small area around (x, y)."""
    points = []
    for i in range(count):
        offset = (i - count // 2) * 2
        points.append((max(0, x + offset), y + rng.randint(-1, 2)))
    return points
