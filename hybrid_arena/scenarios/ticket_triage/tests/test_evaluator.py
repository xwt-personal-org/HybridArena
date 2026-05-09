from __future__ import annotations

from hybrid_arena.scenarios.ticket_triage.evaluator import evaluate_ticket_cases


def test_evaluate_ticket_cases_reports_accuracy_macro_f1_and_unknown_rate() -> None:
    cases = [
        {"ticket_text": "基站弱覆盖导致掉线", "expected_label": "radio_access"},
        {"ticket_text": "核心网注册失败，疑似 AMF 异常", "expected_label": "core_network"},
        {"ticket_text": "用户话费账单异常", "expected_label": "billing"},
    ]

    result = evaluate_ticket_cases(cases)

    assert result.total == 3
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["macro_f1"] == 1.0
    assert result.metrics["unknown_rate"] == 0.0
