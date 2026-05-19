import json

import pytest

from hybrid_arena.inference.llm_planner import LLMPlanner, StubLLMProvider, validate_llm_decision
from hybrid_arena.inference.planner_state import summarize_game_state
from hybrid_arena.minimoba.env import parallel_env


def test_stub_provider_returns_schema_valid_macro_action():
    env = parallel_env(map_size=16, team_size=2, max_steps=20)
    env.reset(seed=7)
    state = summarize_game_state(env.game_state, team="red")
    planner = LLMPlanner(provider=StubLLMProvider("PUSH_LANE"))
    assert planner.plan(state) == "PUSH_LANE"


def test_invalid_llm_decision_rejected():
    with pytest.raises(ValueError, match="Missing LLM decision fields"):
        validate_llm_decision({"macro_action": "GROUP_MID"})


def test_llm_json_bias_fields_must_be_numeric():
    payload = json.loads(StubLLMProvider("GROUP_MID").generate("prompt"))
    payload["reward_bias"] = {"kill": "high"}
    with pytest.raises(ValueError, match="numeric"):
        validate_llm_decision(payload)
