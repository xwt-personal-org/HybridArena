from hybrid_arena.distributed.messages import PolicyVersion, TrajectoryChunk, TrajectoryStep


def test_message_round_trip_preserves_contract():
    chunk = TrajectoryChunk(
        actor_id="actor-0",
        policy_version=PolicyVersion(version=3, updated_at=1.0, metadata={"tag": "x"}),
        steps=[
            TrajectoryStep(
                agent_id="red_0",
                action=[0, 3, 8],
                reward=0.5,
                done=False,
                behavior_log_prob=-0.1,
                behavior_version=3,
                observation_digest="abc",
            )
        ],
    )
    loaded = TrajectoryChunk.from_dict(chunk.to_dict())
    assert loaded == chunk
    assert loaded.size == 1
