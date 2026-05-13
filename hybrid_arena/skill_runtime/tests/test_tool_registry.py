"""Tests for deterministic skill-runtime tool registry."""

from __future__ import annotations

from hybrid_arena.skill_runtime.tool_registry import ToolDescriptor, ToolRegistry


def test_builtin_controller_discovery_returns_static_descriptors() -> None:
    registry = ToolRegistry.discover_builtin_controllers()
    descriptors = registry.list()
    names = {item.name for item in descriptors}

    assert "mock_annotate_formatted" in names
    assert "mock_create_pytest_skeleton" in names
    assert all(item.source == "builtin" for item in descriptors)


def test_registry_register_get_and_preview_are_deterministic() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(
            name="sample_tool",
            description="sample",
            effects=("read_fs",),
            trust="local",
            source="test",
        )
    )

    descriptor = registry.get("sample_tool")
    preview = registry.preview("sample_tool", {"path": "src/app.py"})

    assert descriptor is not None
    assert descriptor.name == "sample_tool"
    assert preview["tool"] == "sample_tool"
    assert preview["would_execute"] is True
    assert preview["event"]["path"] == "src/app.py"


def test_registry_preview_unknown_tool_reports_error() -> None:
    preview = ToolRegistry().preview("missing_tool")

    assert preview["would_execute"] is False
    assert preview["error"] == "unknown_tool"
