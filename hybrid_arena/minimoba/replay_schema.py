"""Offline replay schema for MiniMOBA research datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class OfflineTransition:
    episode_id: str
    step: int
    agent_id: str
    observation: dict[str, Any]
    action: list[int]
    reward: float
    next_observation: dict[str, Any]
    done: bool
    action_mask: list[int]
    global_state: list[float]
    next_global_state: list[float]
    behavior_policy: str = "synthetic_rule_expert"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(_to_serializable(asdict(self)), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OfflineTransition:
        return cls(**payload)


@dataclass
class EpisodeReplay:
    episode_id: str
    transitions: list[OfflineTransition]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.transitions)


class ReplayDatasetWriter:
    """Write deterministic JSONL offline transitions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, transitions: Iterable[OfflineTransition]) -> int:
        count = 0
        with self.path.open("w", encoding="utf-8") as handle:
            for transition in transitions:
                handle.write(transition.to_json() + "\n")
                count += 1
        return count


class ReplayDatasetReader:
    """Read JSONL offline transitions and group them by episode."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[OfflineTransition]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield OfflineTransition.from_dict(json.loads(line))

    def read_all(self) -> list[OfflineTransition]:
        return list(iter(self))

    def episodes(self) -> list[EpisodeReplay]:
        grouped: dict[str, list[OfflineTransition]] = {}
        for transition in self:
            grouped.setdefault(transition.episode_id, []).append(transition)
        return [
            EpisodeReplay(episode_id=episode_id, transitions=items)
            for episode_id, items in sorted(grouped.items())
        ]
