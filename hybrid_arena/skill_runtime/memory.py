"""SQLite memory store for the AgentBench skill runtime."""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hybrid_arena.skill_runtime.workspace import Workspace


@dataclass(frozen=True, init=False)
class SkillMemoryRecord:
    """Durable skill-runtime memory item.

    Canonical fields follow the L2-lite contract.  L0/L1 aliases
    (``skill_id``, ``value``, ``confidence``, ``decay_at``) are accepted to
    keep existing callers stable while the public API moves to namespaces.
    """

    key: str
    namespace: str
    kind: str
    payload: dict[str, Any]
    tags: frozenset[str]
    salience: float
    created_at: float
    updated_at: float
    expires_at: float

    def __init__(
        self,
        *,
        key: str,
        namespace: str = "skill_runtime",
        kind: str = "memory",
        payload: dict[str, Any] | None = None,
        tags: frozenset[str] | set[str] | None = None,
        salience: float = 1.0,
        created_at: float = 0.0,
        updated_at: float = 0.0,
        expires_at: float = 0.0,
        skill_id: str | None = None,
        value: dict[str, Any] | None = None,
        confidence: float | None = None,
        decay_at: float | None = None,
        metadata: dict[str, Any] | None = None,
        successes: int = 0,
        failures: int = 0,
    ) -> None:
        resolved_payload = dict(payload if payload is not None else value or {})
        if metadata:
            resolved_payload.setdefault("metadata", dict(metadata))
        if successes or failures:
            resolved_payload.setdefault("successes", successes)
            resolved_payload.setdefault("failures", failures)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "namespace", skill_id or namespace)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload", resolved_payload)
        object.__setattr__(self, "tags", frozenset(tags or frozenset()))
        object.__setattr__(
            self,
            "salience",
            salience if confidence is None else confidence,
        )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(
            self,
            "expires_at",
            expires_at if decay_at is None else decay_at,
        )

    @property
    def skill_id(self) -> str:
        """Compatibility alias for ``namespace``."""
        return self.namespace

    @property
    def value(self) -> dict[str, Any]:
        """Compatibility alias for ``payload``."""
        return self.payload

    @property
    def confidence(self) -> float:
        """Compatibility alias for ``salience``."""
        return self.salience

    @property
    def decay_at(self) -> float:
        """Compatibility alias for ``expires_at``."""
        return self.expires_at

    @property
    def metadata(self) -> dict[str, Any]:
        """Compatibility metadata view."""
        return dict(self.payload.get("metadata", {}))

    @property
    def successes(self) -> int:
        """Compatibility trace success count."""
        return int(self.payload.get("successes", 0))

    @property
    def failures(self) -> int:
        """Compatibility trace failure count."""
        return int(self.payload.get("failures", 0))


class SkillMemoryStore:
    """Small sqlite3-backed store for skill memories and trace summaries."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_memory (
                    key TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    salience REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (namespace, key)
                )
                """
            )

    def upsert(self, record: SkillMemoryRecord) -> None:
        """Insert or replace a memory record."""
        now = time.time()
        created_at = record.created_at
        updated_at = record.updated_at or now
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skill_memory
                    (key, namespace, kind, payload_json, tags_json, salience,
                     created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.key,
                    record.namespace,
                    record.kind,
                    json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(sorted(record.tags), ensure_ascii=False),
                    record.salience,
                    created_at,
                    updated_at,
                    record.expires_at,
                ),
            )

    def get(self, namespace: str, key: str) -> SkillMemoryRecord | None:
        """Return one memory record, or ``None`` if absent."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT key, namespace, kind, payload_json, tags_json, salience,
                       created_at, updated_at, expires_at
                FROM skill_memory
                WHERE namespace = ? AND key = ?
                """,
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def query(
        self,
        namespace: str | None = None,
        kind: str | None = None,
        tags_superset: set[str] | frozenset[str] | None = None,
        min_salience: float = 0.0,
        limit: int = 50,
        *,
        skill_id: str | None = None,
        key_glob: str | None = None,
        min_confidence: float | None = None,
    ) -> list[SkillMemoryRecord]:
        """Return records matching all supplied predicates."""
        resolved_namespace = namespace if namespace is not None else skill_id
        resolved_min = min_salience if min_confidence is None else min_confidence
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, namespace, kind, payload_json, tags_json, salience,
                       created_at, updated_at, expires_at
                FROM skill_memory
                ORDER BY namespace, key
                """
            ).fetchall()

        required_tags = set(tags_superset or set())
        results: list[SkillMemoryRecord] = []
        for row in rows:
            record = self._row_to_record(row)
            if resolved_namespace is not None and record.namespace != resolved_namespace:
                continue
            if kind is not None and record.kind != kind:
                continue
            if key_glob is not None and not fnmatch.fnmatch(record.key, key_glob):
                continue
            if required_tags and not required_tags.issubset(record.tags):
                continue
            if record.salience < resolved_min:
                continue
            results.append(record)
            if len(results) >= limit:
                break
        return results

    def decay(self, now: float | None = None, factor: float = 0.9) -> int:
        """Apply salience decay to expired records and clear their expiry."""
        current = time.time() if now is None else now
        expired = [
            record
            for record in self.query(limit=10_000)
            if record.expires_at > 0.0 and record.expires_at <= current
        ]
        for record in expired:
            self.upsert(
                replace(
                    record,
                    salience=record.salience * factor,
                    expires_at=0.0,
                    updated_at=current,
                )
            )
        return len(expired)

    def summarize_traces(
        self,
        workspace: Workspace | None = None,
        namespace: str = "skill_runtime",
        *,
        skill_id: str | None = None,
        limit: int = 200,
    ) -> list[SkillMemoryRecord] | dict[str, Any]:
        """Summarize traces into memory records.

        When ``workspace`` is supplied this returns the L2-lite record list.
        Without ``workspace`` it keeps the legacy dict summary used by earlier
        tests against a shared runtime DB.
        """
        if workspace is not None:
            traces = workspace.get_traces(limit=limit)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for trace in traces:
                grouped.setdefault(str(trace["skill_id"]), []).append(trace)
            records: list[SkillMemoryRecord] = []
            now = time.time()
            for skill, skill_traces in sorted(grouped.items()):
                total = len(skill_traces)
                successes = sum(1 for trace in skill_traces if trace["success"])
                records.append(
                    SkillMemoryRecord(
                        namespace=namespace,
                        key=f"trace-summary:{skill}",
                        kind="trace_summary",
                        payload={
                            "skill_id": skill,
                            "total": total,
                            "successes": successes,
                            "failures": total - successes,
                            "success_rate": successes / total if total else 0.0,
                        },
                        tags=frozenset({"trace_summary", skill}),
                        salience=1.0,
                        created_at=now,
                        updated_at=now,
                    )
                )
            return records

        return self._legacy_trace_summary(skill_id=skill_id, limit=limit)

    def _legacy_trace_summary(
        self,
        *,
        skill_id: str | None,
        limit: int,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'traces'"
            ).fetchone()
            if table_exists is None:
                return self._empty_trace_summary(skill_id)
            if skill_id is None:
                rows = conn.execute(
                    """
                    SELECT skill_id, success, residual
                    FROM traces
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT skill_id, success, residual
                    FROM traces
                    WHERE skill_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (skill_id, limit),
                ).fetchall()

        if not rows:
            return self._empty_trace_summary(skill_id)

        total = len(rows)
        successes = sum(1 for row in rows if bool(row[1]))
        failures = total - successes
        avg_residual = sum(float(row[2]) for row in rows) / total
        return {
            "skill_id": skill_id,
            "total": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total,
            "avg_residual": avg_residual,
        }

    @staticmethod
    def _empty_trace_summary(skill_id: str | None) -> dict[str, Any]:
        return {
            "skill_id": skill_id,
            "total": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "avg_residual": 0.0,
        }

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> SkillMemoryRecord:
        return SkillMemoryRecord(
            key=row[0],
            namespace=row[1],
            kind=row[2],
            payload=dict(json.loads(row[3])),
            tags=frozenset(json.loads(row[4])),
            salience=float(row[5]),
            created_at=float(row[6]),
            updated_at=float(row[7]),
            expires_at=float(row[8]),
        )
