"""Tests for objective reward shaping (Phase F13.2-F13.5)."""

from hybrid_arena.minimoba.game_engine import GameState
from hybrid_arena.minimoba.reward_shaper import RewardConfig


def _make_game(team_size=2, map_size=16, max_steps=200, reward_config=None):
    return GameState(
        map_size=map_size,
        team_size=team_size,
        reward_config=reward_config,
        max_steps=max_steps,
        seed=42,
    )


# --- F13.2: _add_team_reward ---


def test_add_team_reward_splits_across_team():
    rc = RewardConfig(objective_enabled=True)
    game = _make_game(reward_config=rc)
    game.reset()
    rewards = dict.fromkeys(game.possible_agents, 0.0)
    game._add_team_reward(rewards, "red", 1.0)
    # Should split 1.0 / 2 = 0.5 per red agent
    assert abs(rewards["red_0"] - 0.5) < 1e-9
    assert abs(rewards["red_1"] - 0.5) < 1e-9
    # Blue should be untouched
    assert rewards["blue_0"] == 0.0
    assert rewards["blue_1"] == 0.0


def test_add_team_reward_ignores_non_positive_amount():
    rc = RewardConfig()
    game = _make_game(reward_config=rc)
    game.reset()
    rewards = dict.fromkeys(game.possible_agents, 0.0)
    game._add_team_reward(rewards, "red", 0.0)
    game._add_team_reward(rewards, "red", -1.0)
    for agent_id in game.possible_agents:
        assert rewards[agent_id] == 0.0


# --- F13.3: objective progress reward in _damage_structure ---


def test_tower_damage_grants_team_objective_reward():
    rc = RewardConfig(
        objective_enabled=True,
        objective_tower_damage_team=0.001,
        objective_step_cap_team=10.0,  # high cap
        damage=0.0,
    )
    game = _make_game(reward_config=rc)
    game.reset()
    # Find a blue tower
    blue_towers = [
        s for s in game.structures.values() if s.team == "blue" and s.structure_type == "tower"
    ]
    assert len(blue_towers) > 0
    tower = blue_towers[0]

    # Simulate red_0 attacking the tower
    attacker = game.heroes["red_0"]
    attacker.x, attacker.y = tower.x, tower.y  # stand on tower
    rewards = dict.fromkeys(game.possible_agents, 0.0)
    game._damage_structure("red_0", attacker, tower, rewards)

    # Tower damage = attack_damage, team reward = attack_damage * 0.001 / 2
    expected_per = attacker.config.attack_damage * 0.001 / 2
    assert rewards["red_0"] > 0
    assert abs(rewards["red_0"] - expected_per) < 1e-6
    assert abs(rewards["red_1"] - expected_per) < 1e-6


def test_base_damage_grants_larger_team_objective_reward_than_tower_damage():
    rc = RewardConfig(
        objective_enabled=True,
        objective_tower_damage_team=0.001,
        objective_base_damage_team=0.003,
        objective_step_cap_team=10.0,
        damage=0.0,
    )
    game = _make_game(reward_config=rc)
    game.reset()

    # Test tower damage reward
    blue_towers = [
        s for s in game.structures.values() if s.team == "blue" and s.structure_type == "tower"
    ]
    tower = blue_towers[0]
    attacker = game.heroes["red_0"]
    attacker.x, attacker.y = tower.x, tower.y
    rewards_tower = dict.fromkeys(game.possible_agents, 0.0)
    game._damage_structure("red_0", attacker, tower, rewards_tower)
    tower_reward = rewards_tower["red_0"]

    # Test base damage reward (on a fresh game with towers destroyed)
    game2 = _make_game(reward_config=rc)
    game2.reset()
    blue_towers2 = [
        s for s in game2.structures.values() if s.team == "blue" and s.structure_type == "tower"
    ]
    for t in blue_towers2:
        t.hp = 0
    game2._sync_structure_counts()

    blue_base = [
        s for s in game2.structures.values() if s.team == "blue" and s.structure_type == "base"
    ]
    base = blue_base[0]
    attacker2 = game2.heroes["red_0"]
    attacker2.x, attacker2.y = base.x, base.y
    rewards_base = dict.fromkeys(game2.possible_agents, 0.0)
    game2._damage_structure("red_0", attacker2, base, rewards_base)
    base_reward = rewards_base["red_0"]

    assert base_reward > tower_reward


def test_objective_step_cap_limits_shared_reward():
    rc = RewardConfig(
        objective_enabled=True,
        objective_tower_damage_team=1.0,  # very high per-damage reward
        objective_step_cap_team=0.1,  # low cap
        damage=0.0,
    )
    game = _make_game(reward_config=rc)
    game.reset()

    blue_towers = [
        s for s in game.structures.values() if s.team == "blue" and s.structure_type == "tower"
    ]
    tower = blue_towers[0]
    attacker = game.heroes["red_0"]
    attacker.x, attacker.y = tower.x, tower.y
    rewards = dict.fromkeys(game.possible_agents, 0.0)
    game._damage_structure("red_0", attacker, tower, rewards)

    # Total team reward should be capped at 0.1
    total = rewards["red_0"] + rewards["red_1"]
    assert total <= 0.1 + 1e-9


# --- F13.4: base exposed one-time reward ---


def test_destroy_last_tower_grants_base_exposed_reward_once():
    rc = RewardConfig(
        objective_enabled=True,
        objective_base_exposed_team=2.0,
        tower=0.0,
        tower_lost=0.0,
        damage=0.0,
        objective_tower_damage_team=0.0,
    )
    game = _make_game(reward_config=rc)
    game.reset()

    blue_towers = [
        s for s in game.structures.values() if s.team == "blue" and s.structure_type == "tower"
    ]
    assert len(blue_towers) == 2

    # Destroy first tower
    t1 = blue_towers[0]
    t1.hp = 1
    attacker = game.heroes["red_0"]
    attacker.x, attacker.y = t1.x, t1.y
    rewards = dict.fromkeys(game.possible_agents, 0.0)
    game._damage_structure("red_0", attacker, t1, rewards)
    # First tower destroyed, but second still alive -> no base exposed reward
    assert rewards["red_0"] == 0.0

    # Destroy second tower
    t2 = blue_towers[1]
    t2.hp = 1
    attacker.x, attacker.y = t2.x, t2.y
    rewards2 = dict.fromkeys(game.possible_agents, 0.0)
    game._damage_structure("red_0", attacker, t2, rewards2)
    # Now all blue towers destroyed -> base exposed reward
    expected_per = 2.0 / 2  # 1.0 per agent
    assert abs(rewards2["red_0"] - expected_per) < 1e-9
    assert abs(rewards2["red_1"] - expected_per) < 1e-9


def test_destroy_non_last_tower_does_not_grant_base_exposed_reward():
    rc = RewardConfig(
        objective_enabled=True,
        objective_base_exposed_team=2.0,
        tower=0.0,
        tower_lost=0.0,
        damage=0.0,
        objective_tower_damage_team=0.0,
    )
    game = _make_game(reward_config=rc)
    game.reset()

    blue_towers = [
        s for s in game.structures.values() if s.team == "blue" and s.structure_type == "tower"
    ]
    t1 = blue_towers[0]
    t1.hp = 1
    attacker = game.heroes["red_0"]
    attacker.x, attacker.y = t1.x, t1.y
    rewards = dict.fromkeys(game.possible_agents, 0.0)
    game._damage_structure("red_0", attacker, t1, rewards)
    # Second tower still alive, no base exposed
    assert rewards["red_0"] == 0.0
    assert rewards["red_1"] == 0.0


# --- F13.5: diagnostic counters ---


def test_structure_damage_counters_update():
    rc = RewardConfig(
        objective_enabled=False,
        damage=0.0,
    )
    game = _make_game(reward_config=rc)
    game.reset()

    blue_towers = [
        s for s in game.structures.values() if s.team == "blue" and s.structure_type == "tower"
    ]
    tower = blue_towers[0]
    attacker = game.heroes["red_0"]
    attacker.x, attacker.y = tower.x, tower.y
    rewards = dict.fromkeys(game.possible_agents, 0.0)
    game._damage_structure("red_0", attacker, tower, rewards)

    assert game.red_tower_damage > 0
    assert game.blue_tower_damage == 0
    assert game.red_base_damage == 0
    assert game.blue_base_damage == 0


def test_structure_damage_counters_reset():
    rc = RewardConfig(objective_enabled=False)
    game = _make_game(reward_config=rc)
    game.reset()
    game.red_tower_damage = 100.0
    game.blue_base_damage = 200.0
    game.reset()
    assert game.red_tower_damage == 0.0
    assert game.blue_base_damage == 0.0
    assert game.base_exposed_rewarded == {"red": False, "blue": False}


# --- Helpers ---


def _find_structure(game, team, structure_type):
    for s in game.structures.values():
        if s.team == team and s.structure_type == structure_type and s.alive:
            return s
    raise ValueError(f"No alive {team} {structure_type} found")
