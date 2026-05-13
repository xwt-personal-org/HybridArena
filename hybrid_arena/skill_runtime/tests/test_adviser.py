from __future__ import annotations

from hybrid_arena.skill_runtime.adviser import SkillRuntimeAdviser
from hybrid_arena.skill_runtime.memory import SkillMemoryRecord, SkillMemoryStore
from hybrid_arena.skill_runtime.sample_skills import build_sample_skills
from hybrid_arena.skill_runtime.security import RuntimePermissionPolicy


def test_adviser_reports_policy_summary_and_disabled_write_effects(tmp_path) -> None:
    adviser = SkillRuntimeAdviser(SkillMemoryStore(tmp_path / "state.db"))

    advice = adviser.advise(skills=build_sample_skills(), policy=RuntimePermissionPolicy())
    advice_by_id = {item["id"]: item for item in advice}

    assert "write_effects_disabled" in advice_by_id
    assert advice_by_id["tool_policy_summary"]["allowed_tool_count"] == 1
    assert advice_by_id["tool_policy_summary"]["blocked_tool_count"] == 4
    assert "WRITE_FS" in advice_by_id["tool_policy_summary"]["blocked_effects"]


def test_adviser_reports_memory_hygiene_findings(tmp_path) -> None:
    store = SkillMemoryStore(tmp_path / "state.db")
    store.upsert(
        SkillMemoryRecord(
            namespace="demo",
            key="stale",
            payload={"note": "old"},
            created_at=10.0,
            updated_at=10.0,
            expires_at=11.0,
        )
    )
    store.record_trace_summary(skill_id="write_summary", success_rate=0.25, attempts=4)
    adviser = SkillRuntimeAdviser(store)

    advice = adviser.advise(skills=build_sample_skills(), policy=RuntimePermissionPolicy(), now=20.0)
    memory_advice = next(item for item in advice if item["id"] == "memory_hygiene")

    assert memory_advice["stale_record_count"] == 1
    assert memory_advice["low_success_skill_count"] == 1
