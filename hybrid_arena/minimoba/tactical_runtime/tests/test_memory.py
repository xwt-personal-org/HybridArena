"""Tests for episode-level tactical memory storage."""

from __future__ import annotations

import sqlite3

from hybrid_arena.minimoba.tactical_runtime.memory import (
    TacticalMemoryRecord,
    TacticalMemoryStore,
)


class TestTacticalMemoryStore:
    """Tests for sqlite-backed tactical memory."""

    def test_record_and_query_round_trip(self, tmp_path):
        db_path = tmp_path / "tactical_memory.sqlite3"
        store = TacticalMemoryStore(db_path)
        store.record(TacticalMemoryRecord(
            episode_id="ep-1",
            agent_id="red_0",
            skill_id="farm_resources",
            success=True,
            reward_delta=1.25,
            tick=7,
            tags=("resource", "lane"),
            state_summary={"target": [4, 7]},
            action={"move": 1, "skill": 0, "target": 0},
        ))

        records = store.query(episode_id="ep-1", agent_id="red_0", skill_id="farm_resources")

        assert len(records) == 1
        record = records[0]
        assert record.episode_id == "ep-1"
        assert record.agent_id == "red_0"
        assert record.skill_id == "farm_resources"
        assert record.success is True
        assert record.reward_delta == 1.25
        assert record.tick == 7
        assert record.tags == frozenset({"resource", "lane"})
        assert record.state_summary == {"target": [4, 7]}
        assert record.action == {"move": 1, "skill": 0, "target": 0}
        store.close()

    def test_query_filters_by_success_and_limit(self, tmp_path):
        store = TacticalMemoryStore(tmp_path / "memory.sqlite3")
        for idx, success in enumerate((True, False, True)):
            store.record(TacticalMemoryRecord(
                episode_id="ep-1",
                agent_id="red_0",
                skill_id="push_objective",
                success=success,
                reward_delta=float(idx),
                tick=idx,
            ))

        records = store.query(skill_id="push_objective", success=True, limit=1)

        assert len(records) == 1
        assert records[0].success is True
        assert records[0].tick == 2
        store.close()

    def test_query_filters_by_tags_superset(self, tmp_path):
        store = TacticalMemoryStore(tmp_path / "memory.sqlite3")
        store.record(TacticalMemoryRecord(
            episode_id="ep-1",
            agent_id="red_0",
            skill_id="farm_resources",
            success=True,
            reward_delta=1.0,
            tags=frozenset({"resource", "lane"}),
        ))

        records = store.query(tags_superset={"resource"})

        assert len(records) == 1
        assert records[0].skill_id == "farm_resources"
        store.close()

    def test_summarize_skill_outcomes(self, tmp_path):
        store = TacticalMemoryStore(tmp_path / "memory.sqlite3")
        store.record(TacticalMemoryRecord(
            episode_id="ep-1",
            agent_id="red_0",
            skill_id="farm_resources",
            success=True,
            reward_delta=1.0,
        ))
        store.record(TacticalMemoryRecord(
            episode_id="ep-1",
            agent_id="red_1",
            skill_id="farm_resources",
            success=False,
            reward_delta=-0.5,
        ))
        store.record(TacticalMemoryRecord(
            episode_id="ep-1",
            agent_id="red_0",
            skill_id="retreat_when_low",
            success=True,
            reward_delta=0.25,
        ))

        summary = store.summarize_skill_outcomes()

        assert summary["farm_resources"] == {
            "attempts": 2,
            "successes": 1,
            "failures": 1,
            "success_rate": 0.5,
            "mean_reward_delta": 0.25,
        }
        assert summary["retreat_when_low"]["attempts"] == 1
        store.close()

    def test_creates_required_indexes(self, tmp_path):
        db_path = tmp_path / "memory.sqlite3"
        store = TacticalMemoryStore(db_path)
        store.close()

        conn = sqlite3.connect(db_path)
        try:
            indexes = {
                row[1] for row in conn.execute("PRAGMA index_list(tactical_memory)")
            }
        finally:
            conn.close()

        assert "idx_tactical_memory_episode_agent_skill" in indexes
        assert "idx_tactical_memory_skill_success" in indexes
