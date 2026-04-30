"""Tests for tower/base objective gameplay."""

import numpy as np

from hybrid_arena.minimoba.env import parallel_env
from hybrid_arena.minimoba.game_engine import GameState
from hybrid_arena.minimoba.objectives import StructureState


def _structures(game_state: GameState, team: str, structure_type: str):
    return [
        structure
        for structure in game_state.structures.values()
        if structure.team == team and structure.structure_type == structure_type
    ]


def _place_for_structure_attack(game_state: GameState, attacker_id: str, structure: StructureState):
    attacker = game_state.heroes[attacker_id]
    attacker.x = max(0, structure.x - 1)
    attacker.y = structure.y
    attacker.config.attack_range = 3
    for hero_id, hero in game_state.heroes.items():
        if hero_id != attacker_id and hero.team != attacker.team:
            hero.x = 0
            hero.y = 0


def test_structure_state_damage():
    structure = StructureState(
        structure_id="blue_tower_0",
        team="blue",
        structure_type="tower",
        x=4,
        y=4,
        max_hp=1200.0,
        hp=100.0,
    )

    assert structure.alive
    assert structure.hp_ratio == 100.0 / 1200.0
    assert structure.take_damage(30.0) == 30.0
    assert structure.hp == 70.0
    assert structure.take_damage(100.0) == 70.0
    assert not structure.alive
    assert structure.hp_ratio == 0.0


def test_game_state_initializes_structures():
    for team_size in (2, 4):
        game_state = GameState(map_size=16, team_size=team_size)
        game_state.reset(seed=42)

        assert len(_structures(game_state, "red", "tower")) == 2
        assert len(_structures(game_state, "blue", "tower")) == 2
        assert len(_structures(game_state, "red", "base")) == 1
        assert len(_structures(game_state, "blue", "base")) == 1


def test_auto_attack_damages_enemy_tower_when_in_range():
    game_state = GameState(map_size=16, team_size=2, fog_of_war=False)
    game_state.reset(seed=42)
    tower = _structures(game_state, "blue", "tower")[0]
    _place_for_structure_attack(game_state, "red_0", tower)
    rewards = dict.fromkeys(game_state.possible_agents, 0.0)

    game_state._execute_attack("red_0", game_state.heroes["red_0"], 0, 8, rewards)

    assert tower.hp < tower.max_hp


def test_tower_destroy_updates_reward_and_counts():
    game_state = GameState(map_size=16, team_size=2, fog_of_war=False)
    game_state.reset(seed=42)
    tower = _structures(game_state, "blue", "tower")[0]
    tower.hp = 1.0
    _place_for_structure_attack(game_state, "red_0", tower)
    rewards = dict.fromkeys(game_state.possible_agents, 0.0)

    game_state._execute_attack("red_0", game_state.heroes["red_0"], 0, 8, rewards)

    assert not tower.alive
    assert game_state.blue_towers == 1
    assert game_state.red_gold == 300
    assert rewards["red_0"] >= game_state.reward_config.tower
    assert rewards["blue_0"] <= game_state.reward_config.tower_lost
    assert rewards["blue_1"] <= game_state.reward_config.tower_lost


def test_kill_updates_team_gold():
    game_state = GameState(map_size=16, team_size=2, fog_of_war=False)
    game_state.reset(seed=42)
    rewards = dict.fromkeys(game_state.possible_agents, 0.0)

    game_state._handle_kill("red_0", game_state.heroes["blue_0"], rewards)

    assert game_state.red_gold > 0
    assert game_state.blue_gold == 0


def test_base_destroy_ends_game_with_winner():
    game_state = GameState(map_size=16, team_size=2, fog_of_war=False)
    game_state.reset(seed=42)
    for tower in _structures(game_state, "blue", "tower"):
        tower.hp = 0.0
    game_state._sync_structure_counts()
    base = _structures(game_state, "blue", "base")[0]
    base.hp = 1.0
    _place_for_structure_attack(game_state, "red_0", base)
    rewards = dict.fromkeys(game_state.possible_agents, 0.0)

    game_state._execute_attack("red_0", game_state.heroes["red_0"], 0, 8, rewards)

    assert game_state.is_game_over()
    assert game_state.get_winner() == "red"


def test_max_steps_tiebreaker_uses_objectives():
    game_state = GameState(map_size=16, team_size=2, max_steps=1)
    game_state.reset(seed=42)
    game_state.red_towers = 2
    game_state.blue_towers = 1
    game_state.step_count = 1

    assert game_state.get_winner() == "red"


def test_info_contains_objective_metrics():
    env = parallel_env(map_size=16, team_size=2, max_steps=20)
    obs, _ = env.reset(seed=42)
    actions = {agent: np.array([0, 3, 8], dtype=np.int64) for agent in env.agents}

    next_obs, _, _, _, infos = env.step(actions)

    assert next_obs["red_0"]["global_info"].shape == (10,)
    for key in (
        "team_gold",
        "enemy_gold",
        "ally_tower_hp",
        "enemy_tower_hp",
        "ally_base_hp",
        "enemy_base_hp",
    ):
        assert key in infos["red_0"]
