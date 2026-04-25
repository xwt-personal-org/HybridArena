"""Human-playable MiniMOBA using keyboard controls.

Controls:
    Arrow keys / WASD — Move
    1 — Auto-attack nearest enemy
    2 — Skill 1
    3 — Skill 2
    Q — Quit
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.env import parallel_env


def main():
    """Run the human-playable demo."""
    if pygame is None:
        print("pygame is required. Install with: pip install pygame")
        return

    try:
        import pygame
    except ImportError:
        print("pygame is required for human play mode. Install with: pip install pygame")
        return

    env = parallel_env(map_size=32, team_size=4, max_steps=2000, render_mode="human")
    obs, _ = env.reset(seed=42)

    random_agent = RandomAgent()
    human_agent = "red_0"
    running = True

    try:
        while running and env.agents:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            # Build actions
            actions = {}
            keys = pygame.key.get_pressed()

            for agent in env.agents:
                if agent == human_agent:
                    # Human-controlled
                    move_dir = _get_move_dir(keys)
                    skill = _get_skill(keys)
                    target = 0  # auto-target nearest
                    actions[agent] = np.array([move_dir, skill, target], dtype=np.int64)
                else:
                    actions[agent] = random_agent.act(obs[agent])

            obs, rewards, terms, truncs, infos = env.step(actions)

            # Render
            env.render()

            # Print stats periodically
            if env.game_state.step_count % 50 == 0:
                hero = env.game_state.heroes[human_agent]
                print(
                    f"Step {env.game_state.step_count} | "
                    f"HP: {hero.hp:.0f}/{hero.max_hp:.0f} | "
                    f"Kills: {hero.kills} | Deaths: {hero.deaths} | "
                    f"Red K: {env.game_state.red_kills} Blue K: {env.game_state.blue_kills}"
                )

            if not env.agents:
                winner = env.game_state.get_winner()
                print(f"Game over! Winner: {winner}")
                break

    finally:
        env.close()
        pygame.quit()


def _get_move_dir(keys) -> int:
    """Convert keyboard state to move direction (0-8)."""
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    if up and right:
        return 2
    if down and right:
        return 4
    if down and left:
        return 6
    if up and left:
        return 8
    if up:
        return 1
    if right:
        return 3
    if down:
        return 5
    if left:
        return 7
    return 0


def _get_skill(keys) -> int:
    """Return skill choice based on key press."""
    if keys[pygame.K_2]:
        return 1  # skill 1
    if keys[pygame.K_3]:
        return 2  # skill 2
    if keys[pygame.K_1] or keys[pygame.K_SPACE]:
        return 0  # auto-attack
    return 3  # no attack


if __name__ == "__main__":
    main()
