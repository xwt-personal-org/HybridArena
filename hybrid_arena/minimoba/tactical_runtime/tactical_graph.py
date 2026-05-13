"""Lightweight in-process tactical relation projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_arena.minimoba.tactical_runtime.memory import TacticalMemoryRecord
from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace


@dataclass(init=False)
class TacticalRelation:
    """A directed tactical relation projected from runtime artifacts."""

    src: str
    relation: str
    dst: str
    weight: float
    metadata: dict[str, Any]

    def __init__(
        self,
        src: str | None = None,
        relation: str | None = None,
        dst: str | None = None,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        relation_type: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.src = src if src is not None else str(source_id)
        self.relation = relation if relation is not None else str(relation_type)
        self.dst = dst if dst is not None else str(target_id)
        self.weight = float(weight)
        self.metadata = dict(metadata if metadata is not None else evidence or {})

    @property
    def source_id(self) -> str:
        """Compatibility alias for ``src``."""
        return self.src

    @property
    def target_id(self) -> str:
        """Compatibility alias for ``dst``."""
        return self.dst

    @property
    def relation_type(self) -> str:
        """Compatibility alias for ``relation``."""
        return self.relation

    @property
    def evidence(self) -> dict[str, Any]:
        """Compatibility alias for ``metadata``."""
        return self.metadata


def annotations_to_relations(
    workspace: BattlefieldWorkspace | list[dict[str, Any]],
) -> list[TacticalRelation]:
    """Project annotations into position-to-tag relations."""
    rows = workspace.export_annotations() if hasattr(workspace, "export_annotations") else workspace
    relations: list[TacticalRelation] = []
    for row in rows:
        position = row.get("position", (0, 0))
        x, y = position
        src = f"position:{int(x)},{int(y)}"
        weight = float(row.get("intensity", 1.0))
        metadata = {"position": (int(x), int(y))}
        for tag in sorted(row.get("tags", ())):
            relations.append(TacticalRelation(
                src=src,
                relation="has_tag",
                dst=f"tag:{tag}",
                weight=weight,
                metadata=dict(metadata),
            ))
    return relations


def memory_to_relations(records: list[TacticalMemoryRecord]) -> list[TacticalRelation]:
    """Project tactical memory records into agent-skill-outcome relations."""
    relations: list[TacticalRelation] = []
    for record in records:
        outcome = "success" if record.success else "failure"
        metadata = {
            "episode_id": record.episode_id,
            "agent_id": record.agent_id,
            "tick": record.tick,
            "reward_delta": record.reward_delta,
        }
        relations.append(TacticalRelation(
            src=f"agent:{record.agent_id}",
            relation="used_skill",
            dst=f"skill:{record.skill_id}",
            weight=1.0,
            metadata=dict(metadata),
        ))
        relations.append(TacticalRelation(
            src=f"skill:{record.skill_id}",
            relation="produced_outcome",
            dst=f"outcome:{outcome}",
            weight=record.reward_delta,
            metadata=dict(metadata),
        ))
        for tag in sorted(record.tags):
            relations.append(TacticalRelation(
                src=f"skill:{record.skill_id}",
                relation="targets_tag",
                dst=f"tag:{tag}",
                weight=1.0,
                metadata=dict(metadata),
            ))
    return relations


def query_relations(
    relations: list[TacticalRelation],
    src: str | None = None,
    relation: str | None = None,
    dst: str | None = None,
    min_weight: float = 0.0,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
    relation_type: str | None = None,
) -> list[TacticalRelation]:
    """Filter relation objects by source, target, type, and weight."""
    resolved_src = src if src is not None else source_id
    resolved_dst = dst if dst is not None else target_id
    resolved_relation = relation if relation is not None else relation_type
    result: list[TacticalRelation] = []
    for item in relations:
        if resolved_src is not None and item.src != resolved_src:
            continue
        if resolved_dst is not None and item.dst != resolved_dst:
            continue
        if resolved_relation is not None and item.relation != resolved_relation:
            continue
        if item.weight < min_weight:
            continue
        result.append(item)
    return result
