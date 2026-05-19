"""Rating systems for QA tournaments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Rating:
    name: str
    value: float = 1000.0


class RatingSystem(Protocol):
    def update(self, player: Rating, opponent: Rating, score: float) -> tuple[Rating, Rating]:
        ...


class EloRatingSystem:
    def __init__(self, k_factor: float = 32.0):
        self.k_factor = k_factor

    @staticmethod
    def expected_score(player: Rating, opponent: Rating) -> float:
        return 1.0 / (1.0 + 10.0 ** ((opponent.value - player.value) / 400.0))

    def update(self, player: Rating, opponent: Rating, score: float) -> tuple[Rating, Rating]:
        expected = self.expected_score(player, opponent)
        player_new = player.value + self.k_factor * (score - expected)
        opponent_new = opponent.value + self.k_factor * ((1.0 - score) - (1.0 - expected))
        return Rating(player.name, player_new), Rating(opponent.name, opponent_new)


class TrueSkillLikeRating:
    """Optional interface placeholder without a hard dependency."""

    def update(self, player: Rating, opponent: Rating, score: float) -> tuple[Rating, Rating]:
        raise NotImplementedError("Install and vet a TrueSkill dependency before enabling this adapter.")
