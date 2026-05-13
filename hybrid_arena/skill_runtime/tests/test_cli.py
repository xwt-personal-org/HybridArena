from __future__ import annotations

from hybrid_arena.scripts.skill_runtime_demo import main


def test_policy_summary_prints_counts_and_blocked_effects(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--root",
            str(tmp_path / "workspace"),
            "--db",
            str(tmp_path / "state.db"),
            "--once",
            "--policy-summary",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "allowed_tools: 1" in output
    assert "blocked_tools: 4" in output
    assert "blocked_effects: LLM_CALL, NETWORK, RUN_SHELL, WRITE_FS" in output


def test_list_tools_marks_default_policy_blocks(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--root",
            str(tmp_path / "workspace"),
            "--db",
            str(tmp_path / "state.db"),
            "--once",
            "--list-tools",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "inspect_workspace allowed=True" in output
    assert "write_summary allowed=False" in output
