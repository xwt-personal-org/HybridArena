"""Tests for run_ablation reward config handling."""

import pytest

from hybrid_arena.scripts.run_ablation import _build_reward_config


def test_run_ablation_rejects_unknown_reward_key():
    cfg = {"reward": {"objective_enabled": True, "objective_base_expose_team": 1.0}}
    with pytest.raises(ValueError, match="Unknown reward config keys"):
        _build_reward_config(cfg)


def test_run_ablation_accepts_valid_reward_keys():
    cfg = {
        "reward": {
            "objective_enabled": True,
            "objective_tower_damage_team": 0.001,
            "objective_base_damage_team": 0.003,
            "objective_base_exposed_team": 1.0,
            "objective_step_cap_team": 0.25,
        }
    }
    rc = _build_reward_config(cfg)
    assert rc is not None
    assert rc.objective_enabled is True
    assert rc.objective_tower_damage_team == 0.001


def test_run_ablation_returns_none_when_no_reward_section():
    cfg = {"experiment": {"name": "test"}}
    assert _build_reward_config(cfg) is None


def test_run_ablation_returns_none_when_reward_section_empty():
    cfg = {"reward": {}}
    assert _build_reward_config(cfg) is None
