from __future__ import annotations

import json

from hybrid_arena.scripts.agentbench_run import main


def test_agentbench_run_writes_ticket_report(tmp_path) -> None:
    input_path = tmp_path / "tickets.jsonl"
    output_path = tmp_path / "ticket_report.json"
    input_path.write_text(
        '{"ticket_text":"基站弱覆盖导致掉线","expected_label":"radio_access"}\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--scenario",
            "ticket_triage",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    markdown_path = output_path.with_suffix(".md")

    assert exit_code == 0
    assert payload["scenario"] == "ticket_triage"
    assert payload["metrics"]["accuracy"] == 1.0
    assert markdown_path.exists()
