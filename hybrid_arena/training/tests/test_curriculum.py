"""Tests for curriculum level management."""

from hybrid_arena.training.curriculum import CurriculumManager


def test_curriculum_advances_after_two_successes():
    manager = CurriculumManager(win_threshold=0.6)

    assert manager.current_level()["map_size"] == 16
    assert not manager.maybe_advance({"win_rate": 0.6})
    assert manager.current_level()["map_size"] == 16
    assert manager.maybe_advance({"win_rate": 0.7})
    assert manager.current_level()["map_size"] == 24


def test_curriculum_to_env_kwargs():
    manager = CurriculumManager()

    kwargs = manager.to_env_kwargs()

    assert kwargs["map_size"] == 16
    assert kwargs["team_size"] == 2
