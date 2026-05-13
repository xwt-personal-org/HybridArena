"""Static deterministic tool registry for skill-runtime controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hybrid_arena.skill_runtime.schema import Effect


@dataclass(frozen=True, init=False)
class ToolDescriptor:
    """Describes one deterministic local controller."""

    id: str
    name: str
    controller: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    effects: frozenset[Effect]
    trust_level: str
    source: str
    description: str = field(default="", compare=True)

    def __init__(
        self,
        *,
        id: str = "",
        name: str,
        controller: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        effects: frozenset[Effect] | tuple[Effect | str, ...] | None = None,
        trust_level: str = "",
        source: str,
        description: str = "",
        trust: str = "",
    ) -> None:
        resolved_id = id or name
        converted_effects: set[Effect] = set()
        for effect in effects or frozenset():
            converted_effects.add(effect if isinstance(effect, Effect) else Effect(effect))
        object.__setattr__(self, "id", resolved_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "controller", controller or name)
        object.__setattr__(self, "input_schema", dict(input_schema or {}))
        object.__setattr__(self, "output_schema", dict(output_schema or {}))
        object.__setattr__(self, "effects", frozenset(converted_effects))
        object.__setattr__(self, "trust_level", trust_level or trust)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "description", description)

    @property
    def trust(self) -> str:
        """Compatibility alias for ``trust_level``."""
        return self.trust_level

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor."""
        return {
            "id": self.id,
            "name": self.name,
            "controller": self.controller,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "effects": sorted(effect.value for effect in self.effects),
            "trust_level": self.trust_level,
            "trust": self.trust_level,
            "source": self.source,
            "description": self.description,
        }


BUILTIN_CONTROLLER_DESCRIPTORS: tuple[ToolDescriptor, ...] = (
    ToolDescriptor(
        id="mock_annotate_formatted",
        name="mock_annotate_formatted",
        controller="mock_annotate_formatted",
        input_schema={"type": "workspace_event", "required": ["path"]},
        output_schema={"type": "annotation_update"},
        effects=frozenset({Effect.WRITE_FS}),
        trust_level="local-deterministic",
        source="builtin",
        description="Annotate a Python file as formatted and passing.",
    ),
    ToolDescriptor(
        id="mock_create_pytest_skeleton",
        name="mock_create_pytest_skeleton",
        controller="mock_create_pytest_skeleton",
        input_schema={"type": "workspace_event", "required": ["path"]},
        output_schema={"type": "test_file"},
        effects=frozenset({Effect.WRITE_FS}),
        trust_level="local-deterministic",
        source="builtin",
        description="Create a minimal pytest skeleton for an annotated source file.",
    ),
    ToolDescriptor(
        id="mock_toggle_to_passing",
        name="mock_toggle_to_passing",
        controller="mock_toggle_to_passing",
        input_schema={"type": "workspace_event"},
        output_schema={"type": "annotation_update"},
        effects=frozenset({Effect.WRITE_FS, Effect.RUN_SHELL}),
        trust_level="local-deterministic",
        source="builtin",
        description="Mark a failing annotation as passing.",
    ),
    ToolDescriptor(
        id="mock_update_imports",
        name="mock_update_imports",
        controller="mock_update_imports",
        input_schema={"type": "rename_event", "required": ["old_name", "new_name", "paths"]},
        output_schema={"type": "file_update"},
        effects=frozenset({Effect.READ_FS, Effect.WRITE_FS}),
        trust_level="local-deterministic",
        source="builtin",
        description="Perform deterministic string replacement for rename imports.",
    ),
    ToolDescriptor(
        id="fallback_planner",
        name="fallback_planner",
        controller="fallback_planner",
        input_schema={"type": "workspace_event"},
        output_schema={"type": "manual_escalation"},
        effects=frozenset(),
        trust_level="manual-review",
        source="builtin",
        description="Represents local escalation when no deterministic skill applies.",
    ),
)


class ToolRegistry:
    """Registry for deterministic local tool descriptors."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}

    def register(self, descriptor: ToolDescriptor) -> None:
        """Register or replace a tool descriptor by id."""
        self._tools[descriptor.id] = descriptor

    def get(self, tool_id: str) -> ToolDescriptor | None:
        """Return a tool descriptor by id or controller name."""
        if tool_id in self._tools:
            return self._tools[tool_id]
        for descriptor in self._tools.values():
            if descriptor.controller == tool_id or descriptor.name == tool_id:
                return descriptor
        return None

    def list(
        self,
        effect: Effect | str | None = None,
        trust_level: str | None = None,
    ) -> list[ToolDescriptor]:
        """Return descriptors in deterministic id order."""
        resolved_effect = Effect(effect) if isinstance(effect, str) else effect
        tools = [self._tools[key] for key in sorted(self._tools)]
        if resolved_effect is not None:
            tools = [tool for tool in tools if resolved_effect in tool.effects]
        if trust_level is not None:
            tools = [tool for tool in tools if tool.trust_level == trust_level]
        return tools

    @classmethod
    def discover_builtin_controllers(cls) -> ToolRegistry:
        """Return a registry populated with built-in sample controllers."""
        registry = cls()
        for descriptor in BUILTIN_CONTROLLER_DESCRIPTORS:
            registry.register(descriptor)
        return registry

    def preview(
        self,
        skill_id: str,
        event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic preview for a tool invocation."""
        descriptor = self.get(skill_id)
        if descriptor is None:
            return {
                "tool": skill_id,
                "would_execute": False,
                "error": "unknown_tool",
            }
        return {
            "tool": descriptor.id,
            "controller": descriptor.controller,
            "would_execute": True,
            "effects": sorted(effect.value for effect in descriptor.effects),
            "trust_level": descriptor.trust_level,
            "trust": descriptor.trust_level,
            "source": descriptor.source,
            "event": dict(event or {}),
        }
