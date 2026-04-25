"""Pygame renderer for MiniMOBA."""

from __future__ import annotations

import numpy as np
import pygame

from hybrid_arena.minimoba.game_engine import GameState
from hybrid_arena.minimoba.map_generator import (
    BLUE_BASE,
    BLUE_TOWER,
    BUSH,
    EMPTY,
    OBSTACLE,
    RED_BASE,
    RED_TOWER,
)

COLORS = {
    EMPTY: (40, 40, 40),
    OBSTACLE: (80, 80, 80),
    BUSH: (34, 139, 34),
    RED_BASE: (180, 30, 30),
    BLUE_BASE: (30, 30, 180),
    RED_TOWER: (200, 60, 60),
    BLUE_TOWER: (60, 60, 200),
    "bg": (20, 20, 20),
    "grid": (50, 50, 50),
    "red_hero": (255, 60, 60),
    "blue_hero": (60, 60, 255),
    "hp_green": (0, 200, 0),
    "hp_red": (200, 0, 0),
    "dead": (100, 100, 100),
    "fog": (0, 0, 0, 180),
    "text": (255, 255, 255),
}

HERO_SHAPES = {"tank": "square", "dps": "triangle", "support": "circle"}


class Renderer:
    """Minimal Pygame renderer for MiniMOBA."""

    def __init__(self, map_size: int = 32, tile_size: int = 20):
        self.map_size = map_size
        self.tile_size = tile_size
        self.width = map_size * tile_size
        self.height = map_size * tile_size
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None

    def _init_pygame(self):
        if not pygame.get_init():
            pygame.init()
        if self._font is None:
            self._font = pygame.font.SysFont("Arial", 10)

    def render_human(self, game_state: GameState):
        self._init_pygame()
        if self._screen is None:
            self._screen = pygame.display.set_mode((self.width, self.height))
            self._clock = pygame.time.Clock()
            pygame.display.set_caption(f"MiniMOBA — Step {game_state.step_count}")

        self._draw_frame(game_state)
        pygame.display.flip()
        if self._clock:
            self._clock.tick(30)

    def render_rgb(self, game_state: GameState) -> np.ndarray:
        self._init_pygame()
        surf = pygame.Surface((self.width, self.height))
        self._draw_frame(game_state, surf)
        arr = pygame.surfarray.array3d(surf)
        return arr.transpose(1, 0, 2)

    def _draw_frame(self, gs: GameState, surface: pygame.Surface | None = None):
        surf = surface or self._screen
        surf.fill(COLORS["bg"])

        # Draw terrain
        for y in range(self.map_size):
            for x in range(self.map_size):
                tile = gs.terrain[y, x]
                color = COLORS.get(tile, COLORS["bg"])
                rect = pygame.Rect(
                    x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size
                )
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, COLORS["grid"], rect, 1)

        # Draw heroes
        for hero_id, hero in gs.heroes.items():
            if not hero.alive:
                continue
            color = COLORS["red_hero"] if hero.team == "red" else COLORS["blue_hero"]
            cx = hero.x * self.tile_size + self.tile_size // 2
            cy = hero.y * self.tile_size + self.tile_size // 2
            radius = self.tile_size // 2 - 2

            shape = HERO_SHAPES.get(hero.config.role, "square")
            if shape == "circle":
                pygame.draw.circle(surf, color, (cx, cy), radius)
            elif shape == "triangle":
                pts = [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)]
                pygame.draw.polygon(surf, color, pts)
            else:
                rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
                pygame.draw.rect(surf, color, rect)

            # HP bar
            bar_w = self.tile_size - 4
            bar_h = 3
            bar_x = hero.x * self.tile_size + 2
            bar_y = hero.y * self.tile_size - 5
            hp_ratio = hero.hp_ratio
            if hp_ratio > 0.5:
                bar_color = COLORS["hp_green"]
            elif hp_ratio > 0.25:
                bar_color = (200, 200, 0)
            else:
                bar_color = COLORS["hp_red"]
            pygame.draw.rect(surf, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(surf, bar_color, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

        # Draw HUD text
        if self._font and surface is None:
            texts = [
                f"Step: {gs.step_count}/{gs.max_steps}",
                f"Red K: {gs.red_kills}  Blue K: {gs.blue_kills}",
                f"Red T: {gs.red_towers}  Blue T: {gs.blue_towers}",
            ]
            for i, txt in enumerate(texts):
                text_surf = self._font.render(txt, True, COLORS["text"])
                surf.blit(text_surf, (5, 5 + i * 15))
