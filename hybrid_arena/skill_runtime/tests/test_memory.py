from __future__ import annotations

from hybrid_arena.skill_runtime.memory import SkillMemoryRecord, SkillMemoryStore


def test_new_record_without_created_at_gets_nonzero_timestamp(tmp_path) -> None:
    store = SkillMemoryStore(tmp_path / "state.db")

    saved = store.upsert(SkillMemoryRecord(namespace="demo", key="note", payload={"value": 1}))

    assert saved.created_at > 0.0
    assert saved.updated_at >= saved.created_at


def test_update_preserves_original_created_at(tmp_path) -> None:
    store = SkillMemoryStore(tmp_path / "state.db")
    original = store.upsert(SkillMemoryRecord(namespace="demo", key="note", payload={"value": 1}))

    updated = store.upsert(SkillMemoryRecord(namespace="demo", key="note", payload={"value": 2}))

    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at
    assert updated.payload == {"value": 2}


def test_explicit_created_at_is_honored(tmp_path) -> None:
    store = SkillMemoryStore(tmp_path / "state.db")

    saved = store.upsert(
        SkillMemoryRecord(namespace="demo", key="explicit", payload={"value": 1}, created_at=123.0)
    )

    assert saved.created_at == 123.0


def test_legacy_alias_fields_still_map_to_record_fields(tmp_path) -> None:
    store = SkillMemoryStore(tmp_path / "state.db")

    saved = store.upsert(
        SkillMemoryRecord(
            skill_id="legacy-skill",
            key="legacy-key",
            value={"answer": 42},
            confidence=0.75,
            decay_at=456.0,
        )
    )
    fetched = store.get("legacy-skill", "legacy-key")

    assert fetched == saved
    assert saved.namespace == "legacy-skill"
    assert saved.payload == {"answer": 42}
    assert saved.salience == 0.75
    assert saved.expires_at == 456.0
