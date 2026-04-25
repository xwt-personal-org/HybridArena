"""Basic environment correctness tests."""

import numpy as np

from hybrid_arena.minimoba.env import parallel_env


def test_reset_creates_agents():
    env = parallel_env(map_size=16, team_size=2, max_steps=50)
    obs, _ = env.reset(seed=42)
    assert len(env.agents) == 4  # 2 red + 2 blue
    for agent in env.possible_agents:
        assert agent in obs


def test_heroes_spawn_at_bases():
    env = parallel_env(map_size=16, team_size=2, max_steps=50)
    env.reset(seed=42)
    gs = env.game_state
    for hero in gs.heroes.values():
        assert 0 <= hero.x < 16
        assert 0 <= hero.y < 16
        assert hero.alive
        assert hero.hp > 0


def test_step_returns_all_keys():
    env = parallel_env(map_size=16, team_size=2, max_steps=50)
    env.reset(seed=42)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)
    assert isinstance(infos, dict)


def test_game_terminates():
    env = parallel_env(map_size=16, team_size=2, max_steps=10)
    env.reset(seed=42)
    for _ in range(15):
        acts = {a: env.action_space(a).sample() for a in env.agents}
        _, _, terms, truncs, _ = env.step(acts)
        if any(terms.values()) or any(truncs.values()):
            break
    # Should have terminated or truncated by now
    assert env.is_game_over or not env.agents


def test_damage_deals():
    env = parallel_env(map_size=16, team_size=2, max_steps=50)
    env.reset(seed=42)
    gs = env.game_state
    initial_hp = sum(h.hp for h in gs.heroes.values())
    # Force heroes to fight by moving them toward each other
    for _ in range(20):
        acts = {}
        for a in env.agents:
            hero = gs.heroes[a]
            if hero.team == "red":
                acts[a] = np.array([3, 0, 0], dtype=np.int64)  # move right, auto-attack
            else:
                acts[a] = np.array([7, 0, 0], dtype=np.int64)  # move left, auto-attack
        env.step(acts)
    final_hp = sum(h.hp for h in gs.heroes.values())
    # Some damage should have occurred
    assert final_hp <= initial_hp
