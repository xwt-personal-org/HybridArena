"""Tests for local skill-runtime protocol envelopes."""

from __future__ import annotations

from hybrid_arena.skill_runtime.dispatcher import DispatchResult
from hybrid_arena.skill_runtime.protocol import (
    EnvelopeKind,
    SkillRuntimeError,
    SkillRuntimeMessage,
)
from hybrid_arena.skill_runtime.schema import WorkspaceEvent


def test_message_roundtrip_preserves_kind_payload_and_metadata() -> None:
    message = SkillRuntimeMessage(
        kind=EnvelopeKind.COMMAND,
        payload={"controller": "mock_annotate_formatted"},
        metadata={"source": "test"},
        trace_id="trace-001",
        created_at=123.0,
    )

    restored = SkillRuntimeMessage.from_json(message.to_json())

    assert restored == message


def test_workspace_event_adapter_roundtrips_event_payload() -> None:
    event = WorkspaceEvent(
        kind="file_save",
        path="src/app.py",
        payload={"size": 128},
        created_at=456.0,
    )

    message = SkillRuntimeMessage.from_workspace_event(event, trace_id="trace-evt")
    restored = message.to_workspace_event()

    assert message.kind is EnvelopeKind.EVENT
    assert message.metadata["path"] == "src/app.py"
    assert restored == event


def test_dispatch_result_adapter_is_json_serializable() -> None:
    result = DispatchResult(
        skill_id="format_on_save",
        action={"success": True},
        escalated=False,
        success=True,
        residual=0.0,
        message="ok",
    )

    message = SkillRuntimeMessage.from_dispatch_result(result, trace_id="trace-res")
    restored = SkillRuntimeMessage.from_json(message.to_json())

    assert restored.kind is EnvelopeKind.RESULT
    assert restored.payload["skill_id"] == "format_on_save"
    assert restored.payload["success"] is True


def test_runtime_error_converts_to_error_envelope() -> None:
    error = SkillRuntimeError(
        code="unknown_controller",
        message="Controller missing",
        details={"controller": "missing"},
    )

    envelope = error.to_message(trace_id="trace-err")

    assert envelope.kind is EnvelopeKind.ERROR
    assert envelope.payload["code"] == "unknown_controller"
    assert envelope.payload["details"]["controller"] == "missing"
