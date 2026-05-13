"""Reflex dispatcher: trigger matching, bid ranking and trace recording."""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hybrid_arena.skill_runtime.body_schema import BodySchema
from hybrid_arena.skill_runtime.memory import SkillMemoryRecord, SkillMemoryStore
from hybrid_arena.skill_runtime.protocol import SkillRuntimeMessage
from hybrid_arena.skill_runtime.schema import Skill, Trigger, WorkspaceEvent
from hybrid_arena.skill_runtime.workspace import Workspace

ControllerFn = Callable[[Skill, WorkspaceEvent, Workspace], dict]
_CONTROLLER_REGISTRY: dict[str, ControllerFn] = {}


def register_controller(name: str, fn: ControllerFn) -> None:
    """Register a deterministic controller by name."""
    _CONTROLLER_REGISTRY[name] = fn


@dataclass(frozen=True)
class DispatchPolicy:
    """Policy knobs for reflex bid scoring."""

    tonic_inhibition: float = 0.0
    repeat_window: int = 5
    failure_penalty: float = 0.1
    escalation_threshold: float = 0.0
    global_annotation_weight: float = 0.5
    no_go_weight: float | None = None


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of a single dispatch cycle.

    Attributes:
        skill_id: Id of the winning skill, or ``None`` if escalated.
        action: Controller-produced action object (``None`` if escalated).
        escalated: ``True`` when no positive bid was placed and fallback ran.
        success: ``True`` when the action executed successfully.
        residual: Remaining error after execution (0.0 = clean).
        message: Human-readable status message.
    """

    skill_id: str | None = None
    action: Any | None = None
    escalated: bool = False
    success: bool = False
    residual: float = 0.0
    message: str = ""


class ReflexDispatcher:
    """Bid-based reactive dispatcher.

    For each incoming :class:`WorkspaceEvent` the dispatcher evaluates all
    registered triggers, computes a bid for each matched skill, selects the
    winner (highest positive bid), executes its controller, and records a
    trace.

    Bid formula::

        bid = trigger_score * skill.salience - tonic_inhibition - no_go_penalty

    where ``tonic_inhibition = 0.0`` (L0) and
    ``no_go_penalty = skill.no_go_traces * 0.1``.

    Args:
        body: The body-schema providing skill access.
        workspace: Backing workspace for trace/event recording.
        fallback_planner: Optional callable invoked when no positive bid
            exists.  Signature: ``(event) -> DispatchResult``.
    """

    def __init__(
        self,
        body: BodySchema,
        workspace: Workspace,
        fallback_planner: Any | None = None,
        policy: DispatchPolicy | None = None,
        memory: SkillMemoryStore | None = None,
    ) -> None:
        self._body = body
        self._workspace = workspace
        self._fallback_planner = fallback_planner
        self._policy = policy or DispatchPolicy()
        self._memory = memory

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_trigger(
        trigger_kind: str,
        trigger_spec: str,
        event: WorkspaceEvent,
        workspace: Workspace,
    ) -> bool:
        """Return ``True`` if the trigger matches the event.

        Supports three trigger kinds:

        * ``glob`` — fnmatch against ``event.path``.
        * ``regex`` — full regex match against ``event.path``.
        * ``annotation`` — workspace query; ``True`` if any paths returned.
        """
        if trigger_kind == "glob":
            return fnmatch.fnmatch(event.path, trigger_spec)
        if trigger_kind == "regex":
            return bool(re.fullmatch(trigger_spec, event.path))
        if trigger_kind == "annotation":
            # spec is the tag or status to check
            if event.path:
                annotation = workspace.get_annotation(event.path)
                if annotation is not None and (
                    trigger_spec in annotation.tags or annotation.status == trigger_spec
                ):
                    return True
            tagged_paths = workspace.query_paths(any_tag=trigger_spec)
            status_paths = workspace.query_paths(status=trigger_spec)
            return bool(tagged_paths or status_paths)
        return False

    def _tonic_inhibition(self, skill: Skill, event: WorkspaceEvent | None = None) -> float:
        """Return the tonic inhibition term for *skill*."""
        return self._policy.tonic_inhibition

    def _no_go_penalty(self, skill: Skill, event: WorkspaceEvent | None = None) -> float:
        """Return the no-go penalty term for *skill*."""
        weight = (
            self._policy.failure_penalty
            if self._policy.no_go_weight is None
            else self._policy.no_go_weight
        )
        return skill.no_go_traces * weight

    def _trigger_score(self, trigger: Trigger, event: WorkspaceEvent) -> float:
        """Return a trigger score for this event, or 0.0 when unmatched."""
        if trigger.kind == "glob":
            return trigger.salience if fnmatch.fnmatch(event.path, trigger.spec) else 0.0
        if trigger.kind == "regex":
            return trigger.salience if re.fullmatch(trigger.spec, event.path) else 0.0
        if trigger.kind == "annotation":
            event_path_match = False
            if event.path:
                annotation = self._workspace.get_annotation(event.path)
                event_path_match = annotation is not None and (
                    trigger.spec in annotation.tags or annotation.status == trigger.spec
                )
            if event_path_match:
                return trigger.salience
            global_match = bool(
                self._workspace.query_paths(any_tag=trigger.spec)
                or self._workspace.query_paths(status=trigger.spec)
            )
            if global_match:
                return trigger.salience * self._policy.global_annotation_weight
        return 0.0

    def tonic_inhibition(self, skill: Skill) -> float:
        """Compatibility wrapper for L0/L1 tests."""
        return self._tonic_inhibition(skill)

    def no_go_penalty(self, skill: Skill) -> float:
        """Compatibility wrapper for L0/L1 tests."""
        return self._no_go_penalty(skill)

    def trigger_score(self, trigger: Trigger, event: WorkspaceEvent) -> float:
        """Compatibility wrapper for L0/L1 tests."""
        return self._trigger_score(trigger, event)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, event: WorkspaceEvent) -> DispatchResult:
        """Match triggers, rank bids, execute winner and record trace.

        Args:
            event: The workspace event to dispatch on.

        Returns:
            A :class:`DispatchResult` describing what happened.
        """
        # Record the event itself.
        self._body.update(event)

        # Evaluate all triggers and collect bids.
        bids: list[tuple[float, Skill, float]] = []  # (bid, skill, trigger_score)
        for skill in self._body.current_affordances(top_k=64):
            for trigger in skill.triggers:
                trigger_score = self._trigger_score(trigger, event)
                if trigger_score <= 0.0:
                    continue
                bid = (
                    trigger_score * skill.salience
                    - self._tonic_inhibition(skill, event)
                    - self._no_go_penalty(skill, event)
                )
                if bid > self._policy.escalation_threshold:
                    bids.append((bid, skill, trigger_score))

        # Select winner (highest bid).
        if bids:
            bids.sort(key=lambda b: b[0], reverse=True)
            _bid, winner, _ts = bids[0]
            return self._execute_skill(winner, event)

        # No positive bid — try fallback planner.
        if self._fallback_planner is not None:
            result = self._fallback_planner(event)
            event_envelope = SkillRuntimeMessage.from_workspace_event(event)
            result_envelope = SkillRuntimeMessage.from_dispatch_result(result)
            self._workspace.record_trace(
                skill_id="fallback",
                event_kind=event.kind,
                input_snapshot={
                    "path": event.path,
                    "kind": event.kind,
                    "envelope": event_envelope.trace_metadata(),
                },
                output_snapshot={
                    "escalated": True,
                    "envelope": result_envelope.trace_metadata(),
                },
                success=result.success,
                residual=result.residual,
            )
            return result

        # Pure escalation.
        result = DispatchResult(
            skill_id=None,
            action=None,
            escalated=True,
            success=False,
            residual=1.0,
            message="No matching skill; escalated.",
        )
        event_envelope = SkillRuntimeMessage.from_workspace_event(event)
        result_envelope = SkillRuntimeMessage.from_dispatch_result(result)
        self._workspace.record_trace(
            skill_id="__escalated__",
            event_kind=event.kind,
            input_snapshot={
                "path": event.path,
                "kind": event.kind,
                "envelope": event_envelope.trace_metadata(),
            },
            output_snapshot={
                "escalated": True,
                "envelope": result_envelope.trace_metadata(),
            },
            success=False,
            residual=1.0,
        )
        return result

    def _execute_skill(self, skill: Skill, event: WorkspaceEvent) -> DispatchResult:
        """Execute the controller for *skill* and record a trace.

        The L0 prototype uses local deterministic controllers registered by
        name. Missing or invalid controllers fail explicitly.
        """
        controller = _CONTROLLER_REGISTRY.get(skill.controller)
        if controller is None:
            action_payload = {
                "error": "Unknown controller",
                "controller": skill.controller,
            }
            success = False
            residual = 1.0
            message = f"Skill '{skill.name}' failed: Unknown controller"
        else:
            try:
                action_payload = controller(skill, event, self._workspace)
                if not isinstance(action_payload, dict):
                    raise TypeError("Controller must return a dict")
                json.dumps(action_payload)
                success = bool(action_payload.get("success", False))
                residual = 0.0 if success else 1.0
                if success:
                    message = str(
                        action_payload.get(
                            "message",
                            f"Executed skill '{skill.name}'.",
                        )
                    )
                else:
                    message = str(
                        action_payload.get(
                            "error",
                            f"Skill '{skill.name}' failed.",
                        )
                    )
            except Exception as exc:
                action_payload = {"error": str(exc), "controller": skill.controller}
                success = False
                residual = 1.0
                message = f"Skill '{skill.name}' failed: {exc}"

        result = DispatchResult(
            skill_id=skill.id,
            action=action_payload,
            escalated=False,
            success=success,
            residual=residual,
            message=message,
        )
        event_envelope = SkillRuntimeMessage.from_workspace_event(event)
        result_envelope = SkillRuntimeMessage.from_dispatch_result(result)
        output_snapshot = dict(action_payload)
        output_snapshot["envelope"] = result_envelope.trace_metadata()
        self._workspace.record_trace(
            skill_id=skill.id,
            event_kind=event.kind,
            input_snapshot={
                "path": event.path,
                "kind": event.kind,
                "envelope": event_envelope.trace_metadata(),
            },
            output_snapshot=output_snapshot,
            success=success,
            residual=residual,
        )
        self._record_memory(skill, event, success, residual, action_payload)

        return result

    def _record_memory(
        self,
        skill: Skill,
        event: WorkspaceEvent,
        success: bool,
        residual: float,
        action_payload: dict,
    ) -> None:
        if self._memory is None:
            return
        key = event.path or event.kind
        self._memory.upsert(
            SkillMemoryRecord(
                skill_id=skill.id,
                key=key,
                value={
                    "event_kind": event.kind,
                    "success": success,
                    "residual": residual,
                    "action": action_payload,
                },
                tags=frozenset({event.kind}),
                confidence=1.0 if success else 0.25,
                successes=1 if success else 0,
                failures=0 if success else 1,
            )
        )
