from __future__ import annotations

import pytest

from hybrid_arena.scenarios.registry import get_runner, list_scenarios


def test_list_scenarios_contains_agentbench_business_scenarios() -> None:
    scenarios = list_scenarios()

    assert scenarios == ["jd_resume_match", "telecom_rag", "ticket_triage"]


def test_get_runner_rejects_unknown_scenario() -> None:
    with pytest.raises(KeyError, match="unknown"):
        get_runner("unknown")
