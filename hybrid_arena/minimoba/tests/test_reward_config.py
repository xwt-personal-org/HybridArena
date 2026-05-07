"""Tests for RewardConfig objective shaping fields (Phase F13.1)."""

from hybrid_arena.minimoba.reward_shaper import DEFAULT_REWARD_CONFIG, RewardConfig


def test_reward_config_objective_defaults():
    rc = RewardConfig()
    assert rc.objective_enabled is False  # default OFF to avoid baseline pollution
    assert rc.objective_tower_damage_team == 0.001
    assert rc.objective_base_damage_team == 0.003
    assert rc.objective_base_exposed_team == 1.0
    assert rc.objective_step_cap_team == 0.25


def test_reward_config_objective_can_enable():
    rc = RewardConfig(objective_enabled=True)
    assert rc.objective_enabled is True
    assert rc.objective_tower_damage_team == 0.001


def test_reward_config_objective_can_disable():
    rc = RewardConfig(objective_enabled=False)
    assert rc.objective_enabled is False
    assert rc.objective_tower_damage_team == 0.001  # fields still have defaults


def test_reward_config_objective_custom():
    rc = RewardConfig(
        objective_enabled=True,
        objective_tower_damage_team=0.005,
        objective_base_damage_team=0.01,
        objective_base_exposed_team=2.0,
        objective_step_cap_team=0.5,
    )
    assert rc.objective_tower_damage_team == 0.005
    assert rc.objective_base_damage_team == 0.01
    assert rc.objective_base_exposed_team == 2.0
    assert rc.objective_step_cap_team == 0.5


def test_default_reward_config_has_no_objective_shaping():
    assert hasattr(DEFAULT_REWARD_CONFIG, "objective_enabled")
    assert DEFAULT_REWARD_CONFIG.objective_enabled is False


def test_default_env_has_no_objective_shaping_unless_enabled():
    """Default RewardConfig must have objective_enabled=False to avoid
    polluting baseline experiments that don't explicitly opt in."""
    rc = RewardConfig()
    assert rc.objective_enabled is False
