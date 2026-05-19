from hybrid_arena.distributed.messages import PolicyVersion, TrajectoryChunk
from hybrid_arena.distributed.replay_queue import BoundedReplayQueue


def _chunk(name: str) -> TrajectoryChunk:
    return TrajectoryChunk(actor_id=name, policy_version=PolicyVersion(version=0), steps=[])


def test_bounded_queue_rejects_when_full():
    queue = BoundedReplayQueue(max_chunks=1, drop_oldest=False)
    assert queue.push(_chunk("a"))
    assert not queue.push(_chunk("b"))
    assert queue.depth == 1
    assert queue.dropped_chunks == 0


def test_bounded_queue_can_drop_oldest_explicitly():
    queue = BoundedReplayQueue(max_chunks=1, drop_oldest=True)
    assert queue.push(_chunk("a"))
    assert queue.push(_chunk("b"))
    assert queue.dropped_chunks == 1
    assert queue.pop().actor_id == "b"
