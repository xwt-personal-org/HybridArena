"""Tests for lightweight tactical relation helpers."""

from __future__ import annotations

from hybrid_arena.minimoba.tactical_runtime.memory import TacticalMemoryRecord
from hybrid_arena.minimoba.tactical_runtime.tactical_graph import (
    TacticalRelation,
    annotations_to_relations,
    memory_to_relations,
    query_relations,
)
from hybrid_arena.minimoba.tactical_runtime.workspace import (
    BattlefieldAnnotation,
    BattlefieldWorkspace,
)


class TestTacticalGraph:
    """Tests for in-process tactical graph relation projection."""

    def test_annotations_to_relations_uses_exported_rows(self):
        workspace = BattlefieldWorkspace(map_size=32)
        workspace.add_annotation(BattlefieldAnnotation(
            position=(4, 5),
            tags={"dangerous", "objective"},
            intensity=0.8,
        ))

        relations = annotations_to_relations(workspace.export_annotations())

        assert TacticalRelation(
            src="position:4,5",
            dst="tag:dangerous",
            relation="has_tag",
            weight=0.8,
            metadata={"position": (4, 5)},
        ) in relations
        assert TacticalRelation(
            src="position:4,5",
            dst="tag:objective",
            relation="has_tag",
            weight=0.8,
            metadata={"position": (4, 5)},
        ) in relations

    def test_memory_to_relations_projects_skill_outcomes(self):
        records = [
            TacticalMemoryRecord(
                episode_id="ep-1",
                agent_id="red_0",
                skill_id="farm_resources",
                success=True,
                reward_delta=1.5,
                tags=frozenset({"resource_soon"}),
            ),
            TacticalMemoryRecord(
                episode_id="ep-1",
                agent_id="red_1",
                skill_id="farm_resources",
                success=False,
                reward_delta=-0.5,
            ),
        ]

        relations = memory_to_relations(records)

        assert relations[0].src == "agent:red_0"
        assert relations[0].dst == "skill:farm_resources"
        assert relations[0].relation == "used_skill"
        produced = next(item for item in relations if item.dst == "outcome:success")
        assert produced.relation == "produced_outcome"
        assert produced.weight == 1.5
        assert any(item.dst == "outcome:failure" for item in relations)
        assert any(item.dst == "tag:resource_soon" for item in relations)

    def test_query_relations_filters_by_source_target_and_type(self):
        relations = [
            TacticalRelation("skill:a", "produced", "outcome:success", 1.0),
            TacticalRelation("skill:a", "uses", "tag:objective", 0.5),
            TacticalRelation("skill:b", "produced", "outcome:success", 1.0),
        ]

        result = query_relations(
            relations,
            src="skill:a",
            dst="outcome:success",
            relation="produced",
        )

        assert result == [relations[0]]
