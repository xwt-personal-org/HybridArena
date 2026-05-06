"""Reward function tests."""

import numpy as np

from hybrid_arena.minimoba.env import parallel_env
from hybrid_arena.minimoba.reward_shaper import DEFAULT_REWARD_CONFIG, RewardConfig


def test_reward_config_defaults():
    cfg = DEFAULT_REWARD_CONFIG
    assert cfg.kill == 1.0
    assert cfg.death == -0.8
    assert cfg.win == 5.0
    assert cfg.lose == -5.0


def test_reward_config_custom():
    cfg = RewardConfig(kill=2.0, win=10.0)
    assert cfg.kill == 2.0
    assert cfg.win == 10.0


def test_rewards_include_time_penalty():
    """Every step must include the time penalty reward."""
    env = parallel_env(map_size=16, team_size=2, max_steps=50)
    env.reset(seed=42)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    for agent in env.possible_agents:
        assert rewards.get(agent, 0.0) < 0, (
            f"Expected negative time penalty for {agent}, got {rewards.get(agent)}"
        )


def test_kill_and_death_rewards():
    """Kill and death rewards propagate correctly when heroes fight at close range."""
    env = parallel_env(map_size=16, team_size=2, max_steps=500)
    env.reset(seed=42)
    gs = env.game_state

    # Place heroes right next to each other
    gs.heroes["red_0"].x, gs.heroes["red_0"].y = 7, 7
    gs.heroes["red_1"].x, gs.heroes["red_1"].y = 8, 7
    gs.heroes["blue_0"].x, gs.heroes["blue_0"].y = 9, 7
    gs.heroes["blue_1"].x, gs.heroes["blue_1"].y = 10, 7
    # Set all HP low enough to kill quickly
    for h in gs.heroes.values():
        h.hp = 120.0
        h.config = h.config  # keep same config

    found_kill = False
    found_death = False

    for _ in range(100):
        acts = {}
        for a in env.agents:
            if a.startswith("red"):
                acts[a] = np.array([0, 0, 0], dtype=np.int64)  # stay, auto-attack
            else:
                acts[a] = np.array([0, 0, 0], dtype=np.int64)  # stay, auto-attack
        _, rewards, terms, _, _ = env.step(acts)

        for a, r in rewards.items():
            if r > 0.9:
                found_kill = True
            if r < -0.6:
                found_death = True

        if found_kill and found_death:
            break

    assert found_kill, "No kill reward (>0.9) was granted in 100 steps of close combat"
    assert found_death, "No death penalty (<-0.6) was granted in 100 steps of close combat"


def test_damage_gives_reward():
    """Dealing damage should give positive reward on top of time penalty."""
    env = parallel_env(map_size=16, team_size=2, max_steps=200)
    env.reset(seed=42)
    gs = env.game_state

    # Place one pair close
    gs.heroes["red_0"].x, gs.heroes["red_0"].y = 7, 7
    gs.heroes["blue_0"].x, gs.heroes["blue_0"].y = 8, 7

    had_damage = False
    for _ in range(50):
        acts = {}
        for a in env.agents:
            acts[a] = np.array([0, 0, 0], dtype=np.int64)
        _, rewards, _, _, _ = env.step(acts)
        for r in rewards.values():
            if r > 0.0:
                had_damage = True
                break
        if had_damage:
            break

    assert had_damage, "No positive damage reward was ever granted"


def test_rewards_in_range():
    """Rewards from actual game steps should remain in reasonable bounds."""
    env = parallel_env(map_size=16, team_size=2, max_steps=30)
    env.reset(seed=42)
    for _ in range(30):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        _, rewards, terms, truncs, _ = env.step(actions)
        for agent, reward in rewards.items():
            assert -10.0 < reward < 10.0, f"Reward out of range for {agent}: {reward}"
        if not env.agents:
            break


def test_win_lose_reward_on_red_win():
    """When red wins, red agents get win reward and blue agents get lose reward."""
    cfg = RewardConfig(win=5.0, lose=-5.0)
    env = parallel_env(map_size=16, team_size=2, max_steps=500, reward_config=cfg)
    env.reset(seed=42)
    gs = env.game_state

    # Force red win by setting game_winner
    gs.game_winner = "red"

    # Step to trigger game over check
    actions = {a: np.array([0, 0, 0], dtype=np.int64) for a in env.agents}
    _, rewards, terms, _, _ = env.step(actions)

    # Check win/lose rewards were applied
    for agent in gs.possible_agents:
        if agent.startswith("red"):
            # Red should get win reward (plus other step rewards)
            assert rewards[agent] > 0, f"Red agent {agent} should get positive reward on win"
        else:
            # Blue should get lose reward (plus other step rewards)
            assert rewards[agent] < 0, f"Blue agent {agent} should get negative reward on lose"


def test_draw_no_win_lose_reward():
    """On draw (timeout), no win/lose reward should be given."""
    cfg = RewardConfig(win=5.0, lose=-5.0, time_penalty=-0.001)
    env = parallel_env(map_size=16, team_size=2, max_steps=5, reward_config=cfg)
    env.reset(seed=42)

    # Run until timeout
    all_rewards = []
    while env.agents:
        actions = {a: np.array([0, 0, 0], dtype=np.int64) for a in env.agents}
        _, rewards, terms, truncs, _ = env.step(actions)
        all_rewards.append(rewards)
        if not env.agents:
            break

    # On draw, win/lose reward should NOT be given
    # Sum rewards for each agent - should only contain step rewards, not win/lose
    for agent in env.game_state.possible_agents:
        total = sum(r.get(agent, 0.0) for r in all_rewards)
        # With max_steps=5, only time_penalty applies, no win/lose
        # time_penalty = -0.001 * 5 = -0.005
        assert total > -1.0, f"Agent {agent} got suspiciously low reward on draw: {total}"
        assert total < 1.0, f"Agent {agent} got suspiciously high reward on draw: {total}"
