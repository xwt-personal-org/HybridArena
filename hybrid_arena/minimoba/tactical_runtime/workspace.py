"""Battlefield workspace: annotations, events, and spatial observation layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BattlefieldAnnotation:
    """A spatial marker on the battlefield with intensity and decay.

    Annotations represent tactical information such as danger zones,
    contested areas, resource locations, and control regions.
    """

    position: tuple[int, int]
    tags: set[str]
    intensity: float = 1.0
    decay_rate: float = 0.05
    created_at: int = 0
    last_decay_tick: int = 0


@dataclass
class GameEvent:
    """An event that occurred during gameplay for dispatch processing.

    Events trigger skill evaluation and workspace updates.
    """

    kind: str
    agent_id: str = ""
    position: tuple[int, int] = (0, 0)
    payload: dict = field(default_factory=dict)
    tick: int = 0


# Tag sets for observation channel mapping
_DANGER_TAGS = frozenset({"dangerous", "enemy_spotted", "contested"})
_OPPORTUNITY_TAGS = frozenset({"resource_soon", "our_control", "objective"})
_CONTROL_POSITIVE_TAGS = frozenset({"our_control"})
_CONTROL_NEGATIVE_TAGS = frozenset({"enemy_control", "contested"})


class BattlefieldWorkspace:
    """In-memory spatial workspace for battlefield annotations and events.

    Stores annotations in a dict keyed by (x, y) for fast spatial queries.
    Provides observation layer generation for integration with RL observations.
    """

    def __init__(self, map_size: int = 32) -> None:
        """Initialize the workspace.

        Args:
            map_size: Side length of the square battlefield map.
        """
        self.map_size = map_size
        # annotations stored as flat list for flexible querying
        self._annotations: list[BattlefieldAnnotation] = []
        self._events: list[GameEvent] = []

    def add_annotation(self, annotation: BattlefieldAnnotation) -> None:
        """Add a battlefield annotation to the workspace.

        Args:
            annotation: The annotation to add.
        """
        self._annotations.append(annotation)

    def query_annotations(
        self,
        position: tuple[int, int],
        radius: int,
        tags: set[str] | None = None,
    ) -> list[BattlefieldAnnotation]:
        """Query annotations within a radius of a position, optionally filtered by tags.

        Args:
            position: Center position (x, y) for the query.
            radius: Search radius (Chebyshev distance).
            tags: If provided, only return annotations that have at least one matching tag.

        Returns:
            List of matching annotations.
        """
        px, py = position
        results: list[BattlefieldAnnotation] = []
        for ann in self._annotations:
            ax, ay = ann.position
            if max(abs(ax - px), abs(ay - py)) <= radius:
                if tags is None or ann.tags & tags:
                    results.append(ann)
        return results

    def decay_annotations(self, current_tick: int) -> None:
        """Reduce annotation intensity and remove annotations with intensity <= 0.

        Args:
            current_tick: Current game tick for age-based decay.
        """
        surviving: list[BattlefieldAnnotation] = []
        for ann in self._annotations:
            delta = current_tick - ann.last_decay_tick
            if delta > 0:
                ann.intensity -= ann.decay_rate * delta
                ann.last_decay_tick = current_tick
            if ann.intensity > 0:
                surviving.append(ann)
        self._annotations = surviving

    def record_event(self, event: GameEvent) -> None:
        """Record a game event in the workspace.

        Args:
            event: The event to record.
        """
        self._events.append(event)

    def to_observation_layer(self, num_channels: int = 3) -> np.ndarray:
        """Generate a spatial observation layer from current annotations.

        Channel mapping:
            - channel 0 (danger): tags in {dangerous, enemy_spotted, contested}
            - channel 1 (opportunity): tags in {resource_soon, our_control, objective}
            - channel 2 (control): +1 for our_control, -1 for enemy_control/contested

        Args:
            num_channels: Number of channels in the output (default 3).

        Returns:
            numpy array of shape (map_size, map_size, num_channels), values in [-1, 1].
        """
        layer = np.zeros((self.map_size, self.map_size, num_channels), dtype=np.float32)
        for ann in self._annotations:
            x, y = ann.position
            if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                continue
            intensity = min(max(ann.intensity, 0.0), 1.0)

            # channel 0: danger
            if ann.tags & _DANGER_TAGS:
                layer[y, x, 0] = max(layer[y, x, 0], intensity)

            # channel 1: opportunity
            if ann.tags & _OPPORTUNITY_TAGS:
                layer[y, x, 1] = max(layer[y, x, 1], intensity)

            # channel 2: control
            if ann.tags & _CONTROL_POSITIVE_TAGS:
                layer[y, x, 2] = max(layer[y, x, 2], intensity)
            if ann.tags & _CONTROL_NEGATIVE_TAGS:
                layer[y, x, 2] = min(layer[y, x, 2], -intensity)

        return layer

    @property
    def annotation_count(self) -> int:
        """Return the number of active annotations."""
        return len(self._annotations)

    @property
    def event_count(self) -> int:
        """Return the number of recorded events."""
        return len(self._events)

    def clear(self) -> None:
        """Remove all annotations and events."""
        self._annotations.clear()
        self._events.clear()
