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
            source_id="annotation:4,5",
            target_id="tag:dangerous",
            relation_type="has_tag",
            weight=0.8,
            evidence={"position": (4, 5)},
        ) in relations
        assert TacticalRelation(
            source_id="annotation:4,5",
            target_id="tag:objective",
            relation_type="has_tag",
            weight=0.8,
            evidence={"position": (4, 5)},
        ) in relations

    def test_memory_to_relations_projects_skill_outcomes(self):
        records = [
            TacticalMemoryRecord(
                episode_id="ep-1",
                agent_id="red_0",
                skill_id="farm_resources",
                success=True,
                reward_delta=1.5,
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

        assert relations[0].source_id == "skill:farm_resources"
        assert relations[0].target_id == "outcome:success"
        assert relations[0].relation_type == "produced"
        assert relations[0].weight == 1.5
        assert relations[1].target_id == "outcome:failure"

    def test_query_relations_filters_by_source_target_and_type(self):
        relations = [
            TacticalRelation("skill:a", "outcome:success", "produced", 1.0),
            TacticalRelation("skill:a", "tag:objective", "uses", 0.5),
            TacticalRelation("skill:b", "outcome:success", "produced", 1.0),
        ]

        result = query_relations(
            relations,
            source_id="skill:a",
            target_id="outcome:success",
            relation_type="produced",
        )

        assert result == [relations[0]]

