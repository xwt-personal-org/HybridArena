"""Bounded in-memory replay queue with explicit backpressure."""

from __future__ import annotations

from collections import deque

from hybrid_arena.distributed.messages import TrajectoryChunk


class BoundedReplayQueue:
    def __init__(self, max_chunks: int, *, drop_oldest: bool = False):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = max_chunks
        self.drop_oldest = drop_oldest
        self._items: deque[TrajectoryChunk] = deque()
        self.dropped_chunks = 0

    @property
    def depth(self) -> int:
        return len(self._items)

    def push(self, chunk: TrajectoryChunk) -> bool:
        if len(self._items) >= self.max_chunks:
            if not self.drop_oldest:
                return False
            self._items.popleft()
            self.dropped_chunks += 1
        self._items.append(chunk)
        return True

    def pop(self) -> TrajectoryChunk | None:
        if not self._items:
            return None
        return self._items.popleft()
