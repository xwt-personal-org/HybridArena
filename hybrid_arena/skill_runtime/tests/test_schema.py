"""Tests for skill-runtime schema dataclasses."""

from __future__ import annotations

from hybrid_arena.skill_runtime.schema import (
    Annotation,
    Effect,
    ForwardModel,
    Skill,
    Trigger,
    TypedSignature,
    WorkspaceEvent,
)


class TestEffect:
    """Verify enum member values."""

    def test_enum_members(self) -> None:
        assert Effect.READ_FS.value == "read_fs"
        assert Effect.WRITE_FS.value == "write_fs"
        assert Effect.RUN_SHELL.value == "run_shell"
        assert Effect.NETWORK.value == "network"
        assert Effect.LLM_CALL.value == "llm_call"

    def test_enum_count(self) -> None:
        assert len(Effect) == 5


class TestTrigger:
    """Verify trigger construction and defaults."""

    def test_defaults(self) -> None:
        t = Trigger(kind="glob", spec="*.py")
        assert t.kind == "glob"
        assert t.spec == "*.py"
        assert t.salience == 1.0

    def test_custom_salience(self) -> None:
        t = Trigger(kind="regex", spec=".*test.*", salience=0.5)
        assert t.salience == 0.5

    def test_frozen(self) -> None:
        t = Trigger(kind="glob", spec="*.py")
        try:
            t.kind = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


class TestForwardModel:
    """Verify forward model defaults."""

    def test_defaults(self) -> None:
        fm = ForwardModel()
        assert fm.expected_artifacts == frozenset()
        assert fm.invariants == ()
        assert fm.success_predicate == ""

    def test_with_values(self) -> None:
        fm = ForwardModel(
            expected_artifacts=frozenset({"out.py"}),
            invariants=("no_syntax_errors",),
            success_predicate="file_exists",
        )
        assert "out.py" in fm.expected_artifacts
        assert fm.invariants == ("no_syntax_errors",)


class TestTypedSignature:
    """Verify typed signature and effect sets."""

    def test_defaults(self) -> None:
        ts = TypedSignature()
        assert ts.input_type == ""
        assert ts.output_type == ""
        assert ts.effects == frozenset()

    def test_with_effects(self) -> None:
        ts = TypedSignature(
            input_type="file_path",
            output_type="annotation",
            effects=frozenset({Effect.READ_FS, Effect.WRITE_FS}),
        )
        assert Effect.READ_FS in ts.effects
        assert Effect.WRITE_FS in ts.effects
        assert len(ts.effects) == 2


class TestSkill:
    """Verify skill construction and defaults."""

    def test_minimal(self) -> None:
        s = Skill(id="s1", name="Test Skill", triggers=())
        assert s.id == "s1"
        assert s.salience == 1.0
        assert s.no_go_traces == 0
        assert s.prior == 0.5
        assert s.preconditions == ()

    def test_with_triggers(self) -> None:
        t = Trigger(kind="glob", spec="*.py")
        s = Skill(id="s2", name="S2", triggers=(t,))
        assert len(s.triggers) == 1
        assert s.triggers[0].kind == "glob"


class TestWorkspaceEvent:
    """Verify workspace event construction."""

    def test_defaults(self) -> None:
        e = WorkspaceEvent(kind="file_save")
        assert e.kind == "file_save"
        assert e.path == ""
        assert e.payload == {}
        assert e.created_at == 0.0

    def test_with_payload(self) -> None:
        e = WorkspaceEvent(kind="test_fail", path="test_foo.py", payload={"line": 42})
        assert e.payload["line"] == 42


class TestAnnotation:
    """Verify annotation defaults and metadata."""

    def test_defaults(self) -> None:
        a = Annotation(path="src/foo.py")
        assert a.path == "src/foo.py"
        assert a.tags == frozenset()
        assert a.status == "unknown"
        assert a.last_skill == ""
        assert a.decay_at == 0.0
        assert a.lineage == ()

    def test_with_metadata(self) -> None:
        a = Annotation(
            path="src/bar.py",
            tags=frozenset({"py", "needs_test"}),
            status="passing",
            last_skill="format_on_save",
            lineage=("format_on_save", "add_test"),
        )
        assert "py" in a.tags
        assert a.status == "passing"
        assert a.lineage == ("format_on_save", "add_test")
