"""Static verification for WEN-90 ISSUE-F13 gate documentation."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_wen90_issue_f13_gate_decision_is_documented():
    text = (REPO_ROOT / "docs" / "issues.md").read_text(encoding="utf-8")

    required_phrases = [
        "ISSUE-F13",
        "https://github.com/xwt-personal-org/HybridArena/pull/15",
        "4d1595c2ddfe4cd5649dec732368064e6b918b6a",
        "hard_win_rate=1.0",
        "base_exposed_rate=1.0",
        "avg_base_damage=2000.0",
        "tower_damage=2400.0",
        "avg_reward_margin=29.39999999999999",
        "avg_length=163.0",
        "terminal_reasons=[base_destroyed, base_destroyed, base_destroyed]",
        "300k-500k",
        "WEN-44",
        "In Review / Done",
        "\u901a\u8fc7",
    ]

    for phrase in required_phrases:
        assert phrase in text
