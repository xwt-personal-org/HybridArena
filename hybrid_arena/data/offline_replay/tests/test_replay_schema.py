from hybrid_arena.data.offline_replay.synthetic_expert import generate_synthetic_expert_replay
from hybrid_arena.minimoba.replay_schema import ReplayDatasetReader, ReplayDatasetWriter


def test_replay_round_trip_is_deterministic(tmp_path):
    transitions = generate_synthetic_expert_replay(episodes=1, steps_per_episode=3, seed=7)
    path = tmp_path / "replay.jsonl"
    assert ReplayDatasetWriter(path).write(transitions) == len(transitions)
    loaded = ReplayDatasetReader(path).read_all()
    assert [t.to_json() for t in loaded] == [t.to_json() for t in transitions]
    assert loaded[0].behavior_policy == "synthetic_rule_expert"
    assert loaded[0].metadata["source"] == "deterministic_rule_synthetic"


def test_reader_groups_episodes(tmp_path):
    transitions = generate_synthetic_expert_replay(episodes=2, steps_per_episode=2, seed=9)
    path = tmp_path / "replay.jsonl"
    ReplayDatasetWriter(path).write(transitions)
    episodes = ReplayDatasetReader(path).episodes()
    assert [episode.episode_id for episode in episodes] == ["synthetic-9-0", "synthetic-9-1"]
    assert all(episode.length > 0 for episode in episodes)
