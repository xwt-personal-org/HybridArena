from __future__ import annotations

from hybrid_arena.scenarios.ticket_triage.classifier import classify_ticket


def test_classify_ticket_detects_radio_access_issue() -> None:
    prediction = classify_ticket("用户反馈基站附近弱覆盖，手机频繁掉线。")

    assert prediction.label == "radio_access"
    assert prediction.confidence > 0
    assert "基站" in prediction.evidence_keywords
    assert prediction.summary


def test_classify_ticket_falls_back_to_unknown() -> None:
    prediction = classify_ticket("用户描述不清，需要进一步确认。")

    assert prediction.label == "unknown"
    assert prediction.confidence == 0.0
