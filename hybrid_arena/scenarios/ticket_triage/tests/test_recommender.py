from __future__ import annotations

from hybrid_arena.scenarios.ticket_triage.recommender import recommend_troubleshooting_steps


def test_recommender_returns_steps_for_known_label() -> None:
    steps = recommend_troubleshooting_steps("transport")

    assert len(steps) >= 3
    assert any("链路" in step for step in steps)


def test_recommender_requests_more_information_for_unknown() -> None:
    steps = recommend_troubleshooting_steps("unknown")

    assert steps == ["补充故障时间、地点、业务类型、影响范围和错误现象后再分诊。"]
