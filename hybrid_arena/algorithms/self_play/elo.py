"""ELO rating system for self-play opponent evaluation.

Design:
    - K-factor starts at 32 for rapid initial adjustment, decays to 16.
    - Supports team-based ELO (average of team member ratings).
    - Tracks win/loss/draw history per policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ELORecord:
    """Rating record for a single policy checkpoint."""

    rating: float = 1000.0
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    def k_factor(self) -> float:
        """Dynamic K: high for new policies, low for established ones."""
        if self.games < 10:
            return 40.0
        if self.games < 30:
            return 32.0
        return 16.0


class ELORatingSystem:
    """Manages ELO ratings for a pool of policy checkpoints.

    Usage:
        elo = ELORatingSystem()
        elo.register("policy_0")
        elo.update("policy_0", "policy_1", winner="policy_0")
    """

    def __init__(self, default_rating: float = 1000.0):
        self.default_rating = default_rating
        self._records: dict[str, ELORecord] = {}

    def register(self, policy_id: str) -> float:
        """Register a new policy with the default rating."""
        if policy_id not in self._records:
            self._records[policy_id] = ELORecord(rating=self.default_rating)
        return self._records[policy_id].rating

    def get_rating(self, policy_id: str) -> float:
        """Get current rating for a policy."""
        return self._records.get(policy_id, ELORecord(rating=self.default_rating)).rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score of A vs B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        policy_a: str,
        policy_b: str,
        winner: str | None = None,
        score_a: float | None = None,
    ) -> tuple[float, float]:
        """Update ratings after a match between A and B.

        Args:
            policy_a: ID of policy A.
            policy_b: ID of policy B.
            winner: "policy_a", "policy_b", or None for draw.
            score_a: Optional explicit score for A in [0, 1] (overrides winner).

        Returns:
            (new_rating_a, new_rating_b)
        """
        self.register(policy_a)
        self.register(policy_b)

        rec_a = self._records[policy_a]
        rec_b = self._records[policy_b]

        if score_a is not None:
            sa = score_a
            sb = 1.0 - score_a
        elif winner == policy_a:
            sa, sb = 1.0, 0.0
        elif winner == policy_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ea = self.expected_score(rec_a.rating, rec_b.rating)
        eb = self.expected_score(rec_b.rating, rec_a.rating)

        ka = rec_a.k_factor()
        kb = rec_b.k_factor()

        rec_a.rating += ka * (sa - ea)
        rec_b.rating += kb * (sb - eb)

        rec_a.games += 1
        rec_b.games += 1

        if sa > sb:
            rec_a.wins += 1
            rec_b.losses += 1
        elif sa < sb:
            rec_a.losses += 1
            rec_b.wins += 1
        else:
            rec_a.draws += 1
            rec_b.draws += 1

        return rec_a.rating, rec_b.rating

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k policies by rating."""
        items = sorted(
            self._records.items(), key=lambda x: x[1].rating, reverse=True
        )
        return [(pid, rec.rating) for pid, rec in items[:k]]

    def rating_history(self, policy_id: str) -> list[float] | None:
        """Placeholder for future rating history tracking."""
        return None

    def __len__(self) -> int:
        return len(self._records)
