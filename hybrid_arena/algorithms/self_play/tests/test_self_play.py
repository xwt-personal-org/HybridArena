"""Tests for self-play, ELO, and curriculum learning."""

from __future__ import annotations

from hybrid_arena.algorithms.self_play.curriculum import (
    CurriculumLevel,
    CurriculumScheduler,
)
from hybrid_arena.algorithms.self_play.elo import ELORatingSystem
from hybrid_arena.algorithms.self_play.manager import SelfPlayManager


class TestELO:
    def test_default_rating(self):
        elo = ELORatingSystem()
        assert elo.get_rating("p1") == 1000.0

    def test_update_winner(self):
        elo = ELORatingSystem()
        elo.update("a", "b", winner="a")
        assert elo.get_rating("a") > 1000.0
        assert elo.get_rating("b") < 1000.0

    def test_update_draw(self):
        elo = ELORatingSystem()
        ra, rb = elo.update("a", "b", winner=None)
        assert abs(ra - 1000.0) < 1.0
        assert abs(rb - 1000.0) < 1.0

    def test_expected_score_symmetry(self):
        elo = ELORatingSystem()
        ea = elo.expected_score(1200, 1000)
        eb = elo.expected_score(1000, 1200)
        assert abs(ea + eb - 1.0) < 1e-6

    def test_top_k(self):
        elo = ELORatingSystem()
        elo.update("strong", "weak", winner="strong")
        elo.update("strong", "weak", winner="strong")
        top = elo.top_k(1)
        assert top[0][0] == "strong"


class TestSelfPlayManager:
    def test_empty_pool_returns_none(self):
        mgr = SelfPlayManager(pool_size=5)
        assert mgr.get_opponent() is None

    def test_pool_addition(self):
        mgr = SelfPlayManager(pool_size=5, win_threshold=0.0)

        def always_win(a, b):
            return 1.0

        added = mgr.maybe_add_to_pool({"w": 1.0}, always_win, policy_id="p0")
        assert added is True
        assert len(mgr) == 1

        added = mgr.maybe_add_to_pool({"w": 2.0}, always_win, policy_id="p1")
        assert added is True
        assert len(mgr) == 2

    def test_pool_size_limit(self):
        mgr = SelfPlayManager(pool_size=3, win_threshold=0.0)

        def always_win(a, b):
            return 1.0

        for i in range(5):
            mgr.maybe_add_to_pool({"i": i}, always_win, policy_id=f"p{i}")

        assert len(mgr) == 3

    def test_elo_leaderboard(self):
        mgr = SelfPlayManager(pool_size=5, win_threshold=0.0)

        def always_win(a, b):
            return 1.0

        for i in range(3):
            mgr.maybe_add_to_pool({"i": i}, always_win, policy_id=f"p{i}")

        board = mgr.get_elo_leaderboard(k=3)
        assert len(board) == 3


class TestCurriculumScheduler:
    def test_initial_level(self):
        sched = CurriculumScheduler()
        assert sched.current_level == 0
        assert sched.level_name == "entry"

    def test_promote_on_high_win_rate(self):
        sched = CurriculumScheduler(window_size=5, promote_threshold=0.8)
        for _ in range(5):
            result = sched.update(1.0)
        assert result == 1
        assert sched.current_level == 1

    def test_demote_on_low_win_rate(self):
        levels = [
            CurriculumLevel("l0", 16, 2, False, "dense", "rule_based", 500),
            CurriculumLevel("l1", 24, 3, True, "dense", "rule_based", 750),
        ]
        sched = CurriculumScheduler(levels=levels, window_size=5, demote_threshold=0.2)
        sched.current_level = 1  # force start at level 1
        sched._win_history = []
        for _ in range(5):
            result = sched.update(0.0)
        assert result == 0
        assert sched.current_level == 0

    def test_env_config_changes_with_level(self):
        sched = CurriculumScheduler()
        cfg0 = sched.get_env_config()
        assert cfg0["map_size"] == 16

        sched.current_level = 2
        cfg2 = sched.get_env_config()
        assert cfg2["map_size"] == 32
        assert cfg2["team_size"] == 4

    def test_reward_density_scaling(self):
        sched = CurriculumScheduler()
        dense_cfg = sched.get_reward_config()
        # Entry level is dense -> farm should be amplified
        assert dense_cfg["farm"] > 0.1

        sched.current_level = 3
        sparse_cfg = sched.get_reward_config()
        # Advanced level is sparse -> farm should be reduced
        assert sparse_cfg["farm"] < dense_cfg["farm"]

    def test_state_dict_roundtrip(self):
        sched = CurriculumScheduler()
        for _ in range(10):
            sched.update(1.0)
        state = sched.state_dict()
        sched2 = CurriculumScheduler()
        sched2.load_state_dict(state)
        assert sched2.current_level == sched.current_level
