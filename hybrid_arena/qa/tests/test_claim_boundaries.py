import json

import pytest

from hybrid_arena.inference.llm_planner import StubLLMProvider, validate_llm_decision
from hybrid_arena.qa.balance_report import write_reports
from hybrid_arena.qa.scenario_matrix import default_scenarios
from hybrid_arena.qa.tournament import run_tournament


def test_default_tournament_is_rule_smoke():
    result = run_tournament(episodes=1, seed=7, scenarios=[default_scenarios()[0]])
    row = result["rows"][0]
    assert result["evaluation_mode"] == "smoke"
    assert result["policy_source"] == "rule_based"
    assert row["evaluation_mode"] == "smoke"
    assert row["policy_source"] == "rule_based"
    assert "trained" not in row["policy_name"]


def test_macro_adapter_smoke_has_explicit_sources(tmp_path):
    result = run_tournament(episodes=1, seed=7, scenarios=[default_scenarios()[1]])
    row = result["rows"][0]
    assert row["planner_source"] == "macro_adapter_smoke"
    assert row["metrics"]["planner_override_rate_source"] == (
        "macro_adapter_decisions_per_policy_decision"
    )
    assert row["metrics"]["illegal_action_rate_source"] == "pre_step_action_mask"
    paths = write_reports(result, tmp_path)
    markdown = (tmp_path / "qa_tournament.md").read_text(encoding="utf-8")
    assert "Evaluation Mode" in markdown
    assert "Policy Source" in markdown
    assert "Planner Source" in markdown
    assert "Claim Boundary" in markdown
    assert "Open Items" in markdown
    assert "current_policy" not in markdown
    assert paths["markdown"].endswith("qa_tournament.md")


def test_invalid_llm_macro_action_raises_not_normalizes():
    payload = json.loads(StubLLMProvider("UNKNOWN_ACTION").generate("prompt"))
    with pytest.raises(ValueError, match="unknown macro action"):
        validate_llm_decision(payload)
