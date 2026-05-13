"""SQLite-backed deterministic memory for skill-runtime diagnostics."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillMemoryRecord:
    namespace: str | None = None
    key: str = ""
    payload: dict[str, Any] | None = None
    salience: float = 1.0
    created_at: float = 0.0
    updated_at: float = 0.0
    expires_at: float | None = None
    record_type: str = "annotation"
    skill_id: InitVar[str | None] = None
    value: InitVar[dict[str, Any] | None] = None
    confidence: InitVar[float | None] = None
    decay_at: InitVar[float | None] = None

    def __post_init__(
        self,
        skill_id: str | None,
        value: dict[str, Any] | None,
        confidence: float | None,
        decay_at: float | None,
    ) -> None:
        namespace = self.namespace or skill_id
        payload = self.payload if self.payload is not None else value
        salience = self.salience if confidence is None else confidence
        expires_at = self.expires_at if decay_at is None else decay_at
        if not namespace:
            raise ValueError("SkillMemoryRecord requires namespace or skill_id.")
        if not self.key:
            raise ValueError("SkillMemoryRecord requires key.")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "payload", dict(payload or {}))
        object.__setattr__(self, "salience", float(salience))
        object.__setattr__(self, "expires_at", expires_at)


@dataclass(frozen=True)
class SkillTraceSummary:
    skill_id: str
    success_rate: float
    attempts: int
    updated_at: float = field(default_factory=time.time)


class SkillMemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_memory (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    salience REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    record_type TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_trace_summary (
                    skill_id TEXT PRIMARY KEY,
                    success_rate REAL NOT NULL,
                    attempts INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def upsert(self, record: SkillMemoryRecord) -> SkillMemoryRecord:
        now = time.time()
        existing = self.get(record.namespace or "", record.key)
        if existing is None:
            created_at = record.created_at or now
        else:
            created_at = record.created_at or existing.created_at
        updated_at = record.updated_at or now
        saved = SkillMemoryRecord(
            namespace=record.namespace,
            key=record.key,
            payload=record.payload,
            salience=record.salience,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=record.expires_at,
            record_type=record.record_type,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skill_memory (
                    namespace, key, payload_json, salience, created_at, updated_at, expires_at, record_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    saved.namespace,
                    saved.key,
                    json.dumps(saved.payload, ensure_ascii=False, sort_keys=True),
                    saved.salience,
                    saved.created_at,
                    saved.updated_at,
                    saved.expires_at,
                    saved.record_type,
                ),
            )
        return saved

    def get(self, namespace: str, key: str) -> SkillMemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT namespace, key, payload_json, salience, created_at, updated_at, expires_at, record_type
                FROM skill_memory
                WHERE namespace = ? AND key = ?
                """,
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def list_records(self) -> list[SkillMemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT namespace, key, payload_json, salience, created_at, updated_at, expires_at, record_type
                FROM skill_memory
                ORDER BY namespace, key
                """
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def stale_records(self, now: float | None = None) -> list[SkillMemoryRecord]:
        cutoff = time.time() if now is None else now
        return [
            record
            for record in self.list_records()
            if record.expires_at is not None and record.expires_at <= cutoff
        ]

    def record_trace_summary(self, skill_id: str, success_rate: float, attempts: int) -> SkillTraceSummary:
        summary = SkillTraceSummary(skill_id=skill_id, success_rate=success_rate, attempts=attempts)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skill_trace_summary (skill_id, success_rate, attempts, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (summary.skill_id, summary.success_rate, summary.attempts, summary.updated_at),
            )
        return summary

    def low_success_trace_summaries(
        self,
        max_success_rate: float = 0.5,
        min_attempts: int = 1,
    ) -> list[SkillTraceSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT skill_id, success_rate, attempts, updated_at
                FROM skill_trace_summary
                WHERE success_rate <= ? AND attempts >= ?
                ORDER BY success_rate ASC, skill_id ASC
                """,
                (max_success_rate, min_attempts),
            ).fetchall()
        return [
            SkillTraceSummary(
                skill_id=row[0],
                success_rate=float(row[1]),
                attempts=int(row[2]),
                updated_at=float(row[3]),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _record_from_row(row: tuple[Any, ...]) -> SkillMemoryRecord:
        namespace, key, payload_json, salience, created_at, updated_at, expires_at, record_type = row
        return SkillMemoryRecord(
            namespace=namespace,
            key=key,
            payload=json.loads(payload_json),
            salience=float(salience),
            created_at=float(created_at),
            updated_at=float(updated_at),
            expires_at=None if expires_at is None else float(expires_at),
            record_type=record_type,
        )

