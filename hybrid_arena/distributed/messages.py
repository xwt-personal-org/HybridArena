"""Message contracts for local Actor/Learner training."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyVersion:
    version: int
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyVersion:
        return cls(**payload)


@dataclass(frozen=True)
class TrajectoryStep:
    agent_id: str
    action: list[int]
    reward: float
    done: bool
    behavior_log_prob: float
    behavior_version: int
    observation_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajectoryStep:
        return cls(**payload)


@dataclass(frozen=True)
class TrajectoryChunk:
    actor_id: str
    policy_version: PolicyVersion
    steps: list[TrajectoryStep]

    @property
    def size(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "policy_version": self.policy_version.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajectoryChunk:
        return cls(
            actor_id=payload["actor_id"],
            policy_version=PolicyVersion.from_dict(payload["policy_version"]),
            steps=[TrajectoryStep.from_dict(item) for item in payload["steps"]],
        )


@dataclass(frozen=True)
class LearnerMetrics:
    update_count: int
    policy_version: int
    actor_fps: float
    queue_depth: int
    dropped_chunks: int
    policy_version_lag: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
