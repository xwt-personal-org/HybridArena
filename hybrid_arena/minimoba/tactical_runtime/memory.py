"""Episode-level tactical memory backed by sqlite3."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TacticalMemoryRecord:
    """A single tactical skill outcome observed during an episode."""

    episode_id: str
    agent_id: str
    skill_id: str
    success: bool
    reward_delta: float
    tick: int = 0
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    record_id: int | None = None


class TacticalMemoryStore:
    """Small sqlite-backed store for episode-level tactical outcomes."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tactical_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                reward_delta REAL NOT NULL,
                tick INTEGER NOT NULL DEFAULT 0,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tactical_memory_episode_agent_skill
            ON tactical_memory (episode_id, agent_id, skill_id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tactical_memory_skill_success
            ON tactical_memory (skill_id, success)
            """
        )
        self._conn.commit()

    def record(self, record: TacticalMemoryRecord) -> int:
        """Persist one tactical outcome and return its row id."""
        cursor = self._conn.execute(
            """
            INSERT INTO tactical_memory (
                episode_id,
                agent_id,
                skill_id,
                success,
                reward_delta,
                tick,
                tags_json,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.episode_id,
                record.agent_id,
                record.skill_id,
                1 if record.success else 0,
                float(record.reward_delta),
                int(record.tick),
                json.dumps(list(record.tags), ensure_ascii=False),
                json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def query(
        self,
        episode_id: str | None = None,
        agent_id: str | None = None,
        skill_id: str | None = None,
        success: bool | None = None,
        limit: int | None = 100,
    ) -> list[TacticalMemoryRecord]:
        """Query tactical memory records with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if episode_id is not None:
            clauses.append("episode_id = ?")
            params.append(episode_id)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if skill_id is not None:
            clauses.append("skill_id = ?")
            params.append(skill_id)
        if success is not None:
            clauses.append("success = ?")
            params.append(1 if success else 0)

        sql = "SELECT * FROM tactical_memory"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY tick DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def summarize_skill_outcomes(self) -> dict[str, dict[str, float | int]]:
        """Aggregate attempts, success rate, and mean reward delta by skill."""
        rows = self._conn.execute(
            """
            SELECT
                skill_id,
                COUNT(*) AS attempts,
                SUM(success) AS successes,
                AVG(reward_delta) AS mean_reward_delta
            FROM tactical_memory
            GROUP BY skill_id
            ORDER BY skill_id
            """
        ).fetchall()

        summary: dict[str, dict[str, float | int]] = {}
        for row in rows:
            attempts = int(row["attempts"])
            successes = int(row["successes"] or 0)
            failures = attempts - successes
            summary[str(row["skill_id"])] = {
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
                "success_rate": successes / attempts if attempts else 0.0,
                "mean_reward_delta": float(row["mean_reward_delta"] or 0.0),
            }
        return summary

    def close(self) -> None:
        """Close the sqlite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _row_to_record(self, row: sqlite3.Row) -> TacticalMemoryRecord:
        return TacticalMemoryRecord(
            episode_id=str(row["episode_id"]),
            agent_id=str(row["agent_id"]),
            skill_id=str(row["skill_id"]),
            success=bool(row["success"]),
            reward_delta=float(row["reward_delta"]),
            tick=int(row["tick"]),
            tags=tuple(json.loads(row["tags_json"])),
            metadata=dict(json.loads(row["metadata_json"])),
            created_at=str(row["created_at"]),
            record_id=int(row["id"]),
        )

    def __enter__(self) -> TacticalMemoryStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

