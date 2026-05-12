"""Lightweight in-process tactical relation projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hybrid_arena.minimoba.tactical_runtime.memory import TacticalMemoryRecord


@dataclass
class TacticalRelation:
    """A directed tactical relation projected from runtime artifacts."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


def annotations_to_relations(rows: list[dict[str, Any]]) -> list[TacticalRelation]:
    """Project annotation export rows into annotation-to-tag relations."""
    relations: list[TacticalRelation] = []
    for row in rows:
        position = row.get("position", (0, 0))
        x, y = position
        source_id = f"annotation:{int(x)},{int(y)}"
        weight = float(row.get("intensity", 1.0))
        evidence = {"position": (int(x), int(y))}
        for tag in sorted(row.get("tags", ())):
            relations.append(TacticalRelation(
                source_id=source_id,
                target_id=f"tag:{tag}",
                relation_type="has_tag",
                weight=weight,
                evidence=dict(evidence),
            ))
    return relations


def memory_to_relations(records: list[TacticalMemoryRecord]) -> list[TacticalRelation]:
    """Project tactical memory records into skill-to-outcome relations."""
    relations: list[TacticalRelation] = []
    for record in records:
        outcome = "success" if record.success else "failure"
        relations.append(TacticalRelation(
            source_id=f"skill:{record.skill_id}",
            target_id=f"outcome:{outcome}",
            relation_type="produced",
            weight=record.reward_delta,
            evidence={
                "episode_id": record.episode_id,
                "agent_id": record.agent_id,
                "tick": record.tick,
            },
        ))
    return relations


def query_relations(
    relations: list[TacticalRelation],
    source_id: str | None = None,
    target_id: str | None = None,
    relation_type: str | None = None,
) -> list[TacticalRelation]:
    """Filter relation objects by source, target, and type."""
    result: list[TacticalRelation] = []
    for relation in relations:
        if source_id is not None and relation.source_id != source_id:
            continue
        if target_id is not None and relation.target_id != target_id:
            continue
        if relation_type is not None and relation.relation_type != relation_type:
            continue
        result.append(relation)
    return result

