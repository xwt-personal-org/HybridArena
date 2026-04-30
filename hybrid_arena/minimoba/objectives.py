"""Runtime state for MiniMOBA map objectives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructureState:
    structure_id: str
    team: str
    structure_type: str
    x: int
    y: int
    max_hp: float
    hp: float
    attack_range: int = 4
    attack_damage: float = 30.0

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def hp_ratio(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return max(self.hp, 0.0) / self.max_hp

    def take_damage(self, amount: float) -> float:
        actual = min(max(amount, 0.0), self.hp)
        self.hp = max(0.0, self.hp - actual)
        return actual
