"""Lightweight checkpoint pool for self-play opponents."""

from __future__ import annotations

import random


class SelfPlayPool:
    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._opponents: list[dict] = []

    def add_checkpoint(self, path: str, metrics: dict | None = None) -> None:
        self._opponents.append({"path": path, "metrics": metrics or {}})
        if len(self._opponents) > self.max_size:
            self._opponents = self._opponents[-self.max_size :]

    def sample_opponent(self, strategy: str = "recent_or_best") -> dict:
        if not self._opponents:
            raise ValueError("self-play pool is empty")
        if strategy == "recent_or_best":
            recent = self._opponents[-1]
            best = max(
                self._opponents,
                key=lambda item: item["metrics"].get("win_rate", 0.0),
            )
            return random.choice([recent, best])
        if strategy == "random":
            return random.choice(self._opponents)
        raise ValueError(f"unknown self-play sampling strategy: {strategy}")

    def list_opponents(self) -> list[dict]:
        return list(self._opponents)
