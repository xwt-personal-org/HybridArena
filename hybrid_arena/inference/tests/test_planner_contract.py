"""Contract tests for the macro-action planner MVP."""

from dataclasses import asdict

import numpy as np

from hybrid_arena.inference.adapter import MacroActionAdapter
from hybrid_arena.inference.llm_planner import DummyLLMClient, LLMPlanner
from hybrid_arena.inference.macro_actions import canonical_macro_action, validate_macro_action
from hybrid_arena.inference.planner_state import summarize_game_state
from hybrid_arena.inference.rule_planner import RulePlanner
from hybrid_arena.minimoba.env import parallel_env


def test_macro_action_validation():
    assert validate_macro_action("GROUP_MID") == "GROUP_MID"
    assert canonical_macro_action("group_mid", allow_aliases=True) == "GROUP_MID"


def test_summarize_game_state_returns_serializable_state():
    env = parallel_env(map_size=16, team_size=2, max_steps=20)
    env.reset(seed=42)

    state = summarize_game_state(env.game_state, team="red")

    assert asdict(state)["team"] == "red"
    assert asdict(state)["objective_summary"]


def test_rule_planner_returns_valid_macro_action():
    env = parallel_env(map_size=16, team_size=2, max_steps=20)
    env.reset(seed=42)
    state = summarize_game_state(env.game_state, team="red")

    action = RulePlanner().plan(state)

    assert validate_macro_action(action) == action


def test_llm_planner_with_dummy_client():
    env = parallel_env(map_size=16, team_size=2, max_steps=20)
    env.reset(seed=42)
    state = summarize_game_state(env.game_state, team="red")

    action = LLMPlanner(DummyLLMClient(), model_name="dummy").plan(state)

    assert action == "GROUP_MID"


def test_macro_action_adapter_outputs_valid_action():
    env = parallel_env(map_size=16, team_size=2, max_steps=20)
    obs, _ = env.reset(seed=42)

    action = MacroActionAdapter("push_nearest_tower").act(obs["red_0"])
    flat = int(action[0] * 36 + action[1] * 9 + action[2])

    assert action.shape == (3,)
    assert action.dtype == np.int64
    assert obs["red_0"]["action_mask"][flat] == 1
