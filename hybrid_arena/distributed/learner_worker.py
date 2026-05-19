"""Local learner loop for distributed smoke tests."""

from __future__ import annotations

from hybrid_arena.distributed.messages import LearnerMetrics, TrajectoryChunk
from hybrid_arena.distributed.policy_sync import PolicyStore
from hybrid_arena.distributed.replay_queue import BoundedReplayQueue


class LocalLearnerWorker:
    def __init__(self, policy_store: PolicyStore):
        self.policy_store = policy_store
        self.update_count = 0

    def consume(self, queue: BoundedReplayQueue, *, actor_fps: float = 0.0) -> LearnerMetrics:
        max_lag = 0
        chunk: TrajectoryChunk | None
        while True:
            chunk = queue.pop()
            if chunk is None:
                break
            max_lag = max(max_lag, self.policy_store.version_lag(chunk.policy_version.version))
            self.update_count += 1
            self.policy_store.publish_update({"source_actor": chunk.actor_id, "chunk_size": chunk.size})
        return LearnerMetrics(
            update_count=self.update_count,
            policy_version=self.policy_store.current.version,
            actor_fps=actor_fps,
            queue_depth=queue.depth,
            dropped_chunks=queue.dropped_chunks,
            policy_version_lag=max_lag,
        )
