"""Data contracts for the AgentBench skill-runtime L0/L1 prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Effect(Enum):
    """Side-effect categories that a skill may trigger."""

    READ_FS = "read_fs"
    WRITE_FS = "write_fs"
    RUN_SHELL = "run_shell"
    NETWORK = "network"
    LLM_CALL = "llm_call"


@dataclass(frozen=True)
class Trigger:
    """Describes a reactive trigger condition for skill activation.

    Attributes:
        kind: Trigger type — one of 'glob', 'regex', 'annotation'.
        spec: Pattern or specification string interpreted by *kind*.
        salience: Base attention weight for this trigger (default 1.0).
    """

    kind: str
    spec: str
    salience: float = 1.0


@dataclass(frozen=True)
class ForwardModel:
    """Predicted outcomes and invariants for a skill execution.

    This version holds descriptive metadata only; runtime callables are
    **not** serialised.

    Attributes:
        expected_artifacts: Set of file paths expected after skill runs.
        invariants: Tuple of human-readable invariant descriptions.
        success_predicate: Expression or label for success condition.
    """

    expected_artifacts: frozenset[str] = frozenset()
    invariants: tuple[str, ...] = ()
    success_predicate: str = ""


@dataclass(frozen=True)
class TypedSignature:
    """Input/output type annotation and effect footprint for a skill.

    Attributes:
        input_type: Human-readable input type label.
        output_type: Human-readable output type label.
        effects: Set of :class:`Effect` values this skill may produce.
    """

    input_type: str = ""
    output_type: str = ""
    effects: frozenset[Effect] = frozenset()


@dataclass(frozen=True)
class Skill:
    """A registered code-engineering skill with triggers, metadata and controller.

    Attributes:
        id: Unique skill identifier.
        name: Human-readable skill name.
        triggers: Tuple of :class:`Trigger` instances.
        salience: Priority weight used in bid ranking.
        no_go_traces: Accumulated failure count (increases bid penalty).
        prior: Bayesian prior probability of success (0.0–1.0).
        forward_model: Optional forward model for predictions.
        precision: Confidence multiplier (0.0–1.0).
        signature: Optional typed signature.
        controller: Identifier for the controller implementation.
        cost_estimate: Estimated execution cost.
        preconditions: Tuple of precondition labels that must all hold.
        repair_skill: Skill id to invoke if this skill fails.
        provenance: Origin / authorship metadata.
    """

    id: str
    name: str
    triggers: tuple[Trigger, ...]
    salience: float = 1.0
    no_go_traces: int = 0
    prior: float = 0.5
    forward_model: ForwardModel | None = None
    precision: float = 1.0
    signature: TypedSignature | None = None
    controller: str = ""
    cost_estimate: float = 0.0
    preconditions: tuple[str, ...] = ()
    repair_skill: str = ""
    provenance: str = ""


@dataclass(frozen=True)
class WorkspaceEvent:
    """An observable event within the workspace.

    Attributes:
        kind: Event type label (e.g. ``file_save``, ``test_fail``).
        path: File-system path associated with the event.
        payload: Arbitrary key-value payload.
        created_at: Unix timestamp (0.0 means unset).
    """

    kind: str
    path: str = ""
    payload: dict = field(default_factory=dict)
    created_at: float = 0.0


@dataclass(frozen=True)
class Annotation:
    """Metadata tag attached to a workspace path.

    Attributes:
        path: File-system path being annotated.
        tags: Set of string tags.
        status: Status label (e.g. ``unknown``, ``passing``, ``failing``).
        last_skill: Id of the last skill that modified this path.
        decay_at: Unix timestamp at which this annotation expires (0 = never).
        lineage: Tuple of skill ids that have touched this path.
    """

    path: str
    tags: frozenset[str] = frozenset()
    status: str = "unknown"
    last_skill: str = ""
    decay_at: float = 0.0
    lineage: tuple[str, ...] = ()
