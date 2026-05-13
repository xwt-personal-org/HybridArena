"""Local protocol envelopes for the skill runtime.

This module intentionally stays in-process and offline.  It gives the runtime
stable message IDs, correlation IDs and error envelopes without implementing an
external ACP/A2A transport.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hybrid_arena.skill_runtime.schema import WorkspaceEvent

PROTOCOL_VERSION = "skill-runtime/v1"


class EnvelopeKind(Enum):
    """Kinds of local skill-runtime envelopes."""

    EVENT = "event"
    COMMAND = "command"
    RESULT = "result"
    ERROR = "error"
    ADVISORY = "advisory"


@dataclass(frozen=True, init=False)
class SkillRuntimeMessage:
    """Versioned local message envelope.

    The canonical fields match the L2-lite contract.  ``payload``,
    ``metadata`` and ``trace_id`` are accepted as compatibility aliases for the
    L0/L1 tests and map onto ``body`` / ``message_id``.
    """

    version: str
    message_id: str
    kind: EnvelopeKind
    source: str
    target: str
    correlation_id: str
    body: dict[str, Any]
    created_at: float
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False, compare=True)

    def __init__(
        self,
        *,
        version: str = PROTOCOL_VERSION,
        message_id: str = "",
        kind: EnvelopeKind = EnvelopeKind.EVENT,
        source: str = "",
        target: str = "",
        correlation_id: str = "",
        body: dict[str, Any] | None = None,
        created_at: float = 0.0,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> None:
        resolved_body = dict(body if body is not None else payload or {})
        resolved_metadata = dict(metadata or {})
        resolved_message_id = message_id or trace_id or str(uuid.uuid4())
        resolved_source = source or str(resolved_metadata.get("source", ""))
        resolved_target = target or str(resolved_metadata.get("target", ""))
        resolved_correlation_id = correlation_id or resolved_message_id
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "message_id", resolved_message_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", resolved_source)
        object.__setattr__(self, "target", resolved_target)
        object.__setattr__(self, "correlation_id", resolved_correlation_id)
        object.__setattr__(self, "body", resolved_body)
        object.__setattr__(self, "created_at", created_at or time.time())
        object.__setattr__(self, "_metadata", resolved_metadata)

    @property
    def payload(self) -> dict[str, Any]:
        """Compatibility alias for ``body``."""
        return self.body

    @property
    def metadata(self) -> dict[str, Any]:
        """Compatibility metadata view used by older callers."""
        metadata = dict(self._metadata)
        if self.source:
            metadata.setdefault("source", self.source)
        if self.target:
            metadata.setdefault("target", self.target)
        return metadata

    @property
    def trace_id(self) -> str:
        """Compatibility alias for ``message_id``."""
        return self.message_id

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable mapping."""
        return {
            "version": self.version,
            "message_id": self.message_id,
            "kind": self.kind.value,
            "source": self.source,
            "target": self.target,
            "correlation_id": self.correlation_id,
            "body": self.body,
            "created_at": self.created_at,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillRuntimeMessage:
        """Create a message from a decoded JSON mapping."""
        kind = EnvelopeKind(data["kind"])
        body = dict(data.get("body", data.get("payload", {})))
        return cls(
            version=str(data.get("version", PROTOCOL_VERSION)),
            message_id=str(data.get("message_id", data.get("trace_id", ""))),
            kind=kind,
            source=str(data.get("source", "")),
            target=str(data.get("target", "")),
            correlation_id=str(data.get("correlation_id", "")),
            body=body,
            created_at=float(data.get("created_at", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        """Encode this envelope as stable JSON."""
        return message_to_json(self)

    @classmethod
    def from_json(cls, raw: str) -> SkillRuntimeMessage:
        """Decode an envelope from JSON."""
        return message_from_json(raw)

    @classmethod
    def from_workspace_event(
        cls,
        event: WorkspaceEvent,
        *,
        trace_id: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "workspace",
        target: str = "dispatcher",
    ) -> SkillRuntimeMessage:
        """Adapt a :class:`WorkspaceEvent` into an event envelope."""
        message = workspace_event_to_message(
            event,
            source=source,
            target=target,
            message_id=trace_id,
        )
        if metadata:
            merged = dict(message._metadata)
            merged.update(metadata)
            object.__setattr__(message, "_metadata", merged)
        return message

    def to_workspace_event(self) -> WorkspaceEvent:
        """Adapt an event envelope back to :class:`WorkspaceEvent`."""
        if self.kind is not EnvelopeKind.EVENT:
            raise ValueError(f"Expected event envelope, got {self.kind.value}")
        return WorkspaceEvent(
            kind=str(self.body["kind"]),
            path=str(self.body.get("path", "")),
            payload=dict(self.body.get("payload", {})),
            created_at=float(self.body.get("created_at", 0.0)),
        )

    @classmethod
    def from_dispatch_result(
        cls,
        result: Any,
        *,
        trace_id: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "dispatcher",
        target: str = "workspace",
    ) -> SkillRuntimeMessage:
        """Adapt a dispatch result into a result envelope."""
        message = dispatch_result_to_message(
            result,
            source=source,
            target=target,
            correlation_id=trace_id,
        )
        if metadata:
            merged = dict(message._metadata)
            merged.update(metadata)
            object.__setattr__(message, "_metadata", merged)
        return message

    def trace_metadata(self) -> dict[str, Any]:
        """Return compact envelope metadata suitable for trace snapshots."""
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "envelope_kind": self.kind.value,
            "source": self.source,
            "target": self.target,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SkillRuntimeError:
    """Structured local runtime error."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_message(
        self,
        *,
        trace_id: str = "",
        source: str = "runtime",
        target: str = "caller",
    ) -> SkillRuntimeMessage:
        """Convert this error to an error envelope."""
        return new_message(
            EnvelopeKind.ERROR,
            source=source,
            target=target,
            body={
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
            correlation_id=trace_id,
        )


def new_message(
    kind: EnvelopeKind,
    source: str,
    target: str,
    body: dict[str, Any],
    correlation_id: str = "",
    *,
    message_id: str = "",
) -> SkillRuntimeMessage:
    """Create a local runtime envelope with stable metadata."""
    resolved_message_id = message_id or str(uuid.uuid4())
    return SkillRuntimeMessage(
        version=PROTOCOL_VERSION,
        message_id=resolved_message_id,
        kind=kind,
        source=source,
        target=target,
        correlation_id=correlation_id or resolved_message_id,
        body=dict(body),
        created_at=time.time(),
    )


def message_to_json(message: SkillRuntimeMessage) -> str:
    """Serialize a message as deterministic JSON."""
    return json.dumps(message.to_dict(), ensure_ascii=False, sort_keys=True)


def message_from_json(raw: str) -> SkillRuntimeMessage:
    """Deserialize a message from JSON."""
    return SkillRuntimeMessage.from_dict(json.loads(raw))


def workspace_event_to_message(
    event: WorkspaceEvent,
    source: str = "workspace",
    target: str = "dispatcher",
    *,
    message_id: str = "",
    correlation_id: str = "",
) -> SkillRuntimeMessage:
    """Adapt a workspace event into a local event envelope."""
    body = {
        "kind": event.kind,
        "path": event.path,
        "payload": dict(event.payload),
        "created_at": event.created_at,
    }
    message = new_message(
        EnvelopeKind.EVENT,
        source=source,
        target=target,
        body=body,
        correlation_id=correlation_id,
        message_id=message_id,
    )
    object.__setattr__(
        message,
        "_metadata",
        {"event_kind": event.kind, "path": event.path},
    )
    return message


def dispatch_result_to_message(
    result: Any,
    source: str = "dispatcher",
    target: str = "workspace",
    correlation_id: str = "",
) -> SkillRuntimeMessage:
    """Adapt a dispatch result into a local result envelope."""
    return new_message(
        EnvelopeKind.RESULT,
        source=source,
        target=target,
        body={
            "skill_id": result.skill_id,
            "action": result.action,
            "escalated": result.escalated,
            "success": result.success,
            "residual": result.residual,
            "message": result.message,
        },
        correlation_id=correlation_id,
    )
