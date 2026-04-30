import json
import tempfile

from hybrid_arena.inference.planner_trace import PlannerTrace
from hybrid_arena.inference.trace_recorder import PlannerTraceRecorder


def sample_trace() -> PlannerTrace:
    return PlannerTrace(
        episode_id="ep_001",
        step=5,
        team="red",
        planner_state={"ally_hp": 0.8, "enemy_hp": 0.3},
        macro_action="push_nearest_tower",
        reward_delta=1.5,
        win=None,
        metadata={"game_tick": 150},
    )


def test_planner_trace_to_dict_json_serializable():
    trace = sample_trace()
    d = trace.to_dict()
    s = json.dumps(d)
    restored = json.loads(s)
    assert restored["episode_id"] == "ep_001"
    assert restored["step"] == 5
    assert restored["macro_action"] == "push_nearest_tower"
    assert restored["reward_delta"] == 1.5
    assert restored["win"] is None


def test_trace_recorder_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/traces.jsonl"
        recorder = PlannerTraceRecorder(path)
        recorder.add(sample_trace())
        recorder.add(sample_trace())
        recorder.flush()

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "episode_id" in data
            assert "macro_action" in data


def test_trace_recorder_creates_parent_dir():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/nested/dir/traces.jsonl"
        recorder = PlannerTraceRecorder(path)
        recorder.add(sample_trace())
        recorder.flush()

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
