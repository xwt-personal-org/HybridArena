"""SQLite-backed storage for AgentBench runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from hybrid_arena.core.schema import TaskRunResult


class AgentBenchStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_runs_scenario ON task_runs(scenario, created_at)"
            )

    def save_run(self, result: TaskRunResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_runs (
                    run_id, task_id, scenario, output_json, metrics_json, trace_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.task_id,
                    result.scenario,
                    json.dumps(result.output, ensure_ascii=False),
                    json.dumps(result.metrics, ensure_ascii=False),
                    json.dumps(result.trace.to_dict(), ensure_ascii=False),
                ),
            )

    def get_run(self, run_id: str) -> TaskRunResult | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, task_id, scenario, output_json, metrics_json, trace_json
                FROM task_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    def list_runs(self, scenario: str | None = None, limit: int = 50) -> list[TaskRunResult]:
        with self._connect() as conn:
            if scenario is None:
                rows = conn.execute(
                    """
                    SELECT run_id, task_id, scenario, output_json, metrics_json, trace_json
                    FROM task_runs
                    ORDER BY created_at DESC, run_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT run_id, task_id, scenario, output_json, metrics_json, trace_json
                    FROM task_runs
                    WHERE scenario = ?
                    ORDER BY created_at DESC, run_id DESC
                    LIMIT ?
                    """,
                    (scenario, limit),
                ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _row_to_result(row: tuple[str, str, str, str, str, str]) -> TaskRunResult:
        run_id, task_id, scenario, output_json, metrics_json, trace_json = row
        return TaskRunResult.from_dict(
            {
                "run_id": run_id,
                "task_id": task_id,
                "scenario": scenario,
                "output": json.loads(output_json),
                "metrics": json.loads(metrics_json),
                "trace": json.loads(trace_json),
            }
        )
