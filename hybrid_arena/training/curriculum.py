"""Curriculum level manager for MiniMOBA training."""

from __future__ import annotations

from pathlib import Path

import yaml


class CurriculumManager:
    def __init__(
        self,
        config_path: str | Path | None = None,
        win_threshold: float | None = None,
    ):
        self.config_path = Path(config_path) if config_path else self._default_config_path()
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.levels = list(data.get("curriculum", {}).get("levels", []))
        if not self.levels:
            raise ValueError("curriculum levels are missing")
        self.win_threshold = (
            win_threshold
            if win_threshold is not None
            else float(data.get("self_play", {}).get("win_threshold", 0.55))
        )
        self.level_idx = 0
        self._success_streak = 0

    def _default_config_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

    def current_level(self) -> dict:
        return dict(self.levels[self.level_idx])

    def maybe_advance(self, metrics: dict) -> bool:
        if metrics.get("win_rate", 0.0) >= self.win_threshold:
            self._success_streak += 1
        else:
            self._success_streak = 0

        if self._success_streak >= 2 and self.level_idx < len(self.levels) - 1:
            self.level_idx += 1
            self._success_streak = 0
            return True
        return False

    def to_env_kwargs(self) -> dict:
        level = self.current_level()
        return {
            "map_size": level["map_size"],
            "team_size": level["team_size"],
        }
