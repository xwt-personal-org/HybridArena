"""Tests for skill-runtime demo CLI L2-lite options."""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_arena.scripts.skill_runtime_demo import main


def test_cli_lists_tools(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "runtime.db"),
            "--once",
            "--list-tools",
        ]
    )

    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert any(item["name"] == "mock_annotate_formatted" for item in out["tools"])


def test_cli_explains_affordances(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "runtime.db"),
            "--once",
            "--explain-affordances",
        ]
    )

    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert isinstance(out, list)
    assert any(item["id"] == "format_on_save" for item in out)


def test_cli_dispatches_event_from_json(tmp_path: Path, capsys) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"kind": "file_save", "path": "src/app.py", "payload": {}}),
        encoding="utf-8",
    )

    rc = main(
        [
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "runtime.db"),
            "--once",
            "--event-json",
            str(event_path),
        ]
    )

    out = capsys.readouterr().out

    assert rc == 0
    assert "Selected skill: format_on_save" in out
