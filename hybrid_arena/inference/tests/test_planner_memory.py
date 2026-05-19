from hybrid_arena.inference.planner_memory import PlannerMemory, PlannerMemoryEntry


def test_planner_memory_keeps_recent_entries():
    memory = PlannerMemory(max_entries=2)
    memory.add(PlannerMemoryEntry(step=1, team="red", macro_action="GROUP_MID", outcome={}))
    memory.add(PlannerMemoryEntry(step=2, team="red", macro_action="PUSH_LANE", outcome={}))
    memory.add(PlannerMemoryEntry(step=3, team="red", macro_action="RETREAT_FARM", outcome={}))
    recent = memory.summarize(limit=5)
    assert [entry["step"] for entry in recent] == [2, 3]
