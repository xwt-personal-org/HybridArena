"""Local policy version store."""

from __future__ import annotations

from hybrid_arena.distributed.messages import PolicyVersion


class PolicyStore:
    def __init__(self) -> None:
        self._current = PolicyVersion(version=0)

    @property
    def current(self) -> PolicyVersion:
        return self._current

    def publish_update(self, metadata: dict | None = None) -> PolicyVersion:
        self._current = PolicyVersion(version=self._current.version + 1, metadata=metadata or {})
        return self._current

    def version_lag(self, actor_version: int) -> int:
        return max(0, self._current.version - actor_version)
