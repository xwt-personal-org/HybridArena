"""SQLite-backed workspace for the AgentBench skill-runtime."""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import time
from pathlib import Path

from hybrid_arena.skill_runtime.schema import Annotation


class Workspace:
    """Persistent workspace backed by a local SQLite database.

    Stores path annotations, event log, execution traces and body-schema
    snapshots.  All collection fields (tags, lineage, payload) are serialised
    as JSON strings inside SQLite.

    Args:
        root: Project root directory.
        db_path: Path to the SQLite database file (created if absent).
    """

    def __init__(self, root: Path, db_path: Path) -> None:
        self.root = Path(root)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    path TEXT PRIMARY KEY,
                    tags TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'unknown',
                    last_skill TEXT NOT NULL DEFAULT '',
                    decay_at REAL NOT NULL DEFAULT 0.0,
                    lineage TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL DEFAULT '',
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    input_snapshot TEXT NOT NULL DEFAULT '{}',
                    output_snapshot TEXT NOT NULL DEFAULT '{}',
                    success INTEGER NOT NULL DEFAULT 0,
                    residual REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS body_schema_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def annotate(
        self,
        path: str,
        tags: set[str] | None = None,
        status: str = "unknown",
        last_skill: str = "",
        decay_at: float | None = None,
        lineage: tuple[str, ...] | None = None,
    ) -> None:
        """Insert or replace an annotation for *path*.

        Args:
            path: Workspace-relative file path.
            tags: Set of string tags (default empty).
            status: Status label.
            last_skill: Id of the skill that last touched this path.
            decay_at: Optional expiry timestamp.
            lineage: Tuple of skill ids that have modified this path.
        """
        tags = tags or set()
        lineage = lineage or ()
        decay = decay_at if decay_at is not None else 0.0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO annotations
                    (path, tags, status, last_skill, decay_at, lineage)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    json.dumps(sorted(tags)),
                    status,
                    last_skill,
                    decay,
                    json.dumps(list(lineage)),
                ),
            )

    def get_annotation(self, path: str) -> Annotation | None:
        """Return the annotation for *path*, or ``None`` if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT path, tags, status, last_skill, decay_at, lineage "
                "FROM annotations WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            return None
        return Annotation(
            path=row[0],
            tags=frozenset(json.loads(row[1])),
            status=row[2],
            last_skill=row[3],
            decay_at=row[4],
            lineage=tuple(json.loads(row[5])),
        )

    def query_paths(
        self,
        *,
        path_glob: str | None = None,
        tags_superset: set[str] | None = None,
        status: str | None = None,
        any_tag: str | None = None,
    ) -> list[str]:
        """Return paths whose annotations match the given predicate.

        When multiple predicates are supplied, paths must match all of them.
        Calling without predicates preserves the L0 behavior and returns an
        empty list.

        Args:
            path_glob: Path must match this glob pattern.
            tags_superset: Path must have **all** of these tags.
            status: Path must have this exact status.
            any_tag: Path must contain this tag.
        """
        if (
            path_glob is None
            and tags_superset is None
            and status is None
            and any_tag is None
        ):
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path, tags, status FROM annotations"
            ).fetchall()

        result: list[str] = []
        required_tags = tags_superset or set()
        for path, tags_raw, row_status in rows:
            tags = set(json.loads(tags_raw))
            if path_glob is not None and not fnmatch.fnmatch(path, path_glob):
                continue
            if required_tags and not required_tags.issubset(tags):
                continue
            if status is not None and row_status != status:
                continue
            if any_tag is not None and any_tag not in tags:
                continue
            result.append(path)
        return result

    def snapshot_annotations(self) -> list[Annotation]:
        """Return all annotations in deterministic path order."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT path, tags, status, last_skill, decay_at, lineage
                FROM annotations
                ORDER BY path
                """
            ).fetchall()
        return [
            Annotation(
                path=row[0],
                tags=frozenset(json.loads(row[1])),
                status=row[2],
                last_skill=row[3],
                decay_at=row[4],
                lineage=tuple(json.loads(row[5])),
            )
            for row in rows
        ]

    def prune_expired_annotations(self, now: float | None = None) -> int:
        """Delete annotations whose ``decay_at`` timestamp has elapsed."""
        current = time.time() if now is None else now
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM annotations WHERE decay_at > 0.0 AND decay_at <= ?",
                (current,),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def record_event(
        self,
        kind: str,
        path: str = "",
        payload: dict | None = None,
    ) -> None:
        """Append an event to the event log.

        Args:
            kind: Event type label.
            path: Optional file-system path.
            payload: Optional key-value payload.
        """
        payload = payload or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (kind, path, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (kind, path, json.dumps(payload), time.time()),
            )

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def record_trace(
        self,
        skill_id: str,
        event_kind: str,
        input_snapshot: dict,
        output_snapshot: dict,
        success: bool,
        residual: float,
    ) -> None:
        """Record a skill execution trace.

        Args:
            skill_id: Id of the executed skill.
            event_kind: Kind of the triggering event.
            input_snapshot: JSON-serialisable input state.
            output_snapshot: JSON-serialisable output state.
            success: Whether execution succeeded.
            residual: Remaining error after execution.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces
                    (skill_id, event_kind, input_snapshot, output_snapshot,
                     success, residual, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_id,
                    event_kind,
                    json.dumps(input_snapshot),
                    json.dumps(output_snapshot),
                    int(success),
                    residual,
                    time.time(),
                ),
            )

    def get_traces(
        self,
        skill_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent traces, optionally filtered by *skill_id*.

        Args:
            skill_id: If provided, only return traces for this skill.
            limit: Maximum number of rows to return.
        """
        with self._connect() as conn:
            if skill_id is not None:
                rows = conn.execute(
                    """
                    SELECT id, skill_id, event_kind, input_snapshot,
                           output_snapshot, success, residual, created_at
                    FROM traces
                    WHERE skill_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (skill_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, skill_id, event_kind, input_snapshot,
                           output_snapshot, success, residual, created_at
                    FROM traces
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [
            {
                "id": r[0],
                "skill_id": r[1],
                "event_kind": r[2],
                "input_snapshot": json.loads(r[3]),
                "output_snapshot": json.loads(r[4]),
                "success": bool(r[5]),
                "residual": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Body-schema snapshots
    # ------------------------------------------------------------------

    def save_body_snapshot(self, snapshot_json: str) -> None:
        """Persist a body-schema snapshot.

        Args:
            snapshot_json: JSON string of the snapshot dict.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO body_schema_snapshots (snapshot, created_at)
                VALUES (?, ?)
                """,
                (snapshot_json, time.time()),
            )
