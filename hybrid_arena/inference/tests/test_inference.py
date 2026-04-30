"""Tests for inference components."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_arena.inference.llm_planner import LLMPlanner, PlannerStateMachine
from hybrid_arena.inference.strategy_bridge import StrategyToRLBridge


class TestLLMPlanner:
    def test_mock_plan(self):
        planner = LLMPlanner(mode="mock")
        summary = "当前局势 (第100步)\n我方击杀:5 敌方击杀:2 经济差:+1000\n我方队伍状态\n- tank(red_0): 存活, 血量充足, 蓝量充足, 位置(16,16), 技能1就绪, 技能2就绪\n可见敌方 (4人)\n建筑: 我方塔3座存活, 敌方塔2座存活"
        plan = planner.plan(summary)
        assert "strategy" in plan
        assert "assignments" in plan
        assert "target_positions" in plan

    def test_parse_strategy_fallback(self):
        planner = LLMPlanner(mode="mock")
        bad = "some garbage text without json"
        result = planner._parse_strategy(bad)
        assert result["strategy"] == "发育"

    def test_parse_strategy_extract(self):
        planner = LLMPlanner(mode="mock")
        text = 'blah {"strategy":"团战","reasoning":"test"} blah'
        result = planner._parse_strategy(text)
        assert result["strategy"] == "团战"


class TestPlannerStateMachine:
    def test_step(self):
        planner = LLMPlanner(mode="mock")
        sm = PlannerStateMachine(planner, reflect_interval=5)
        summary = "当前局势 (第10步)\n我方击杀:1 敌方击杀:1\n"
        plan = sm.step(summary)
        assert sm.state.turn_count == 1
        assert "strategy" in plan

    def test_history_accumulates(self):
        planner = LLMPlanner(mode="mock")
        sm = PlannerStateMachine(planner)
        for i in range(3):
            sm.step(f"Step {i}")
        assert len(sm.state.action_history) == 3


class TestStrategyToRLBridge:
    def test_reward_modifiers(self):
        bridge = StrategyToRLBridge()
        mods = bridge.get_reward_modifiers("团战")
        assert mods["kill"] == 2.0
        assert mods["farm"] == 0.3

        mods2 = bridge.get_reward_modifiers("发育")
        assert mods2["farm"] == 3.0
        assert mods2["kill"] == 0.3

    def test_apply_reward_shaping(self):
        bridge = StrategyToRLBridge()
        base = {"kill": 1.0, "farm": 0.5, "death": -0.8}
        shaped = bridge.apply_reward_shaping("团战", base)
        assert shaped["kill"] == 2.0
        assert shaped["farm"] == 0.15

    def test_goal_positions(self):
        bridge = StrategyToRLBridge()
        assignments = {
            "target_positions": {
                "tank": [16, 16],
                "dps_1": [10, 10],
            }
        }
        goals = bridge.get_goal_positions(assignments)
        assert "tank" in goals
        assert np.allclose(goals["tank"], [16, 16])

    def test_navigation_bonus(self):
        bridge = StrategyToRLBridge()
        bonus = bridge.navigation_bonus(
            np.array([0.0, 0.0]), np.array([10.0, 0.0]), alpha=0.01
        )
        assert bonus == pytest.approx(-0.1, abs=1e-6)
