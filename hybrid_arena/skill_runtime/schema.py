"""Typed contracts for the deterministic skill runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

PROTOCOL_VERSION = "skill-runtime/1"


class Effect(str, Enum):
    READ_FS = "READ_FS"
    WRITE_FS = "WRITE_FS"
    RUN_SHELL = "RUN_SHELL"
    NETWORK = "NETWORK"
    LLM_CALL = "LLM_CALL"


@dataclass(frozen=True)
class SkillSignature:
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    effects: tuple[Effect, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "effects": [effect.value for effect in self.effects],
        }


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    signature: SkillSignature

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "signature": self.signature.to_dict(),
        }


@dataclass(frozen=True)
class WorkspaceEvent:
    kind: str
    path: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceEvent:
        if not isinstance(data, dict):
            raise ValueError("Workspace event must be a dictionary.")
        kind = data.get("kind")
        path = data.get("path")
        payload = data.get("payload", {})
        if not isinstance(kind, str) or not kind:
            raise ValueError("Workspace event requires a non-empty kind.")
        if not isinstance(path, str) or not path:
            raise ValueError("Workspace event requires a non-empty path.")
        if not isinstance(payload, dict):
            raise ValueError("Workspace event payload must be a dictionary.")
        return cls(kind=kind, path=path, payload=dict(payload))


@dataclass(frozen=True)
class SkillRuntimeMessage:
    kind: str
    body: dict[str, Any]
    version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "kind": self.kind, "body": dict(self.body)}

