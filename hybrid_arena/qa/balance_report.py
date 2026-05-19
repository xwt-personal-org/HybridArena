"""Report writers for QA tournament outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_reports(result: dict, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "qa_tournament.json"
    csv_path = output_dir / "qa_tournament.csv"
    md_path = output_dir / "qa_tournament.md"

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "policy_name",
                "opponent_name",
                "win_rate",
                "hard_win_rate",
                "base_exposed_rate",
                "avg_base_damage",
                "avg_tower_damage",
                "avg_episode_length",
                "illegal_action_rate",
                "planner_override_rate",
                "p50_inference_ms",
                "p95_inference_ms",
                "rating_after",
                "gate_passed",
            ],
        )
        writer.writeheader()
        for row in result["rows"]:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "policy_name": row["policy_name"],
                    "opponent_name": row["opponent_name"],
                    "win_rate": metrics.get("win_rate", 0.0),
                    "hard_win_rate": metrics.get("hard_win_rate", 0.0),
                    "base_exposed_rate": metrics.get("base_exposed_rate", 0.0),
                    "avg_base_damage": metrics.get("avg_base_damage", 0.0),
                    "avg_tower_damage": metrics.get("avg_tower_damage", 0.0),
                    "avg_episode_length": metrics.get("avg_episode_length", 0.0),
                    "illegal_action_rate": metrics.get("illegal_action_rate", 0.0),
                    "planner_override_rate": metrics.get("planner_override_rate", 0.0),
                    "p50_inference_ms": metrics.get("p50_inference_ms", 0.0),
                    "p95_inference_ms": metrics.get("p95_inference_ms", 0.0),
                    "rating_after": row["rating_after"],
                    "gate_passed": row["gate_passed"],
                }
            )

    lines = [
        "# QA Tournament Report",
        "",
        f"- Rating system: {result['rating_system']}",
        f"- Final rating: {result['final_rating']:.2f}",
        "",
        "| Scenario | Win | Hard Win | Base Exposed | Tower Damage | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in result["rows"]:
        metrics = row["metrics"]
        lines.append(
            "| {scenario} | {win:.3f} | {hard:.3f} | {base:.3f} | {tower:.3f} | {gate} |".format(
                scenario=row["scenario"],
                win=metrics.get("win_rate", 0.0),
                hard=metrics.get("hard_win_rate", 0.0),
                base=metrics.get("base_exposed_rate", 0.0),
                tower=metrics.get("avg_tower_damage", 0.0),
                gate="PASS" if row["gate_passed"] else "FAIL",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
