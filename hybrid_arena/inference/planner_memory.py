"""Episodic planner memory for macro decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlannerMemoryEntry:
    step: int
    team: str
    macro_action: str
    outcome: dict[str, float]
    reflection: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlannerMemory:
    def __init__(self, max_entries: int = 16):
        self.max_entries = max_entries
        self._entries: list[PlannerMemoryEntry] = []

    def add(self, entry: PlannerMemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def recent(self, limit: int = 5) -> list[PlannerMemoryEntry]:
        return self._entries[-limit:]

    def summarize(self, limit: int = 5) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.recent(limit)]
