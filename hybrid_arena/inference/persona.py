"""Planner persona metadata hooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PersonaMetadata:
    name: str = "default"
    risk_tolerance: float = 0.5
    coordination_style: str = "balanced"

    def to_dict(self) -> dict:
        return asdict(self)
