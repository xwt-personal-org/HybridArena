"""Tests for the battlefield workspace: annotations, events, observation layers."""

from __future__ import annotations

import numpy as np

from hybrid_arena.minimoba.tactical_runtime.workspace import (
    BattlefieldAnnotation,
    BattlefieldWorkspace,
    GameEvent,
)


class TestBattlefieldAnnotation:
    """Tests for BattlefieldAnnotation dataclass."""

    def test_defaults(self):
        ann = BattlefieldAnnotation(position=(5, 10), tags={"danger"})
        assert ann.position == (5, 10)
        assert ann.tags == {"danger"}
        assert ann.intensity == 1.0
        assert ann.decay_rate == 0.05
        assert ann.created_at == 0
        assert ann.last_decay_tick == 0


class TestGameEvent:
    """Tests for GameEvent dataclass."""

    def test_defaults(self):
        ev = GameEvent(kind="test")
        assert ev.kind == "test"
        assert ev.agent_id == ""
        assert ev.position == (0, 0)
        assert ev.payload == {}
        assert ev.tick == 0


class TestBattlefieldWorkspace:
    """Tests for BattlefieldWorkspace."""

    def test_add_and_count(self):
        ws = BattlefieldWorkspace(map_size=32)
        assert ws.annotation_count == 0
        ws.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"danger"}))
        assert ws.annotation_count == 1

    def test_query_annotations_by_position(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"danger"}))
        ws.add_annotation(BattlefieldAnnotation(position=(20, 20), tags={"danger"}))
        ws.add_annotation(BattlefieldAnnotation(position=(6, 6), tags={"resource"}))

        # Within radius 2 of (5, 5)
        results = ws.query_annotations(position=(5, 5), radius=2)
        assert len(results) == 2  # (5,5) and (6,6)

    def test_query_annotations_by_tags(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"danger"}))
        ws.add_annotation(BattlefieldAnnotation(position=(6, 6), tags={"resource"}))
        ws.add_annotation(BattlefieldAnnotation(position=(7, 7), tags={"danger", "resource"}))

        # Only "resource" tags within radius 5 of (5, 5)
        results = ws.query_annotations(position=(5, 5), radius=5, tags={"resource"})
        assert len(results) == 2  # (6,6) and (7,7)

    def test_decay_annotations_removes_zero_intensity(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(
            position=(5, 5), tags={"danger"}, intensity=1.0, decay_rate=0.1, created_at=0
        ))
        ws.add_annotation(BattlefieldAnnotation(
            position=(10, 10), tags={"resource"}, intensity=0.5, decay_rate=0.1, created_at=0
        ))

        # At tick 5: intensity = 1.0 - 0.1*5 = 0.5 and 0.5 - 0.1*5 = 0.0
        ws.decay_annotations(current_tick=5)
        assert ws.annotation_count == 1  # second one removed

    def test_decay_annotations_preserves_nonzero(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(
            position=(5, 5), tags={"danger"}, intensity=2.0, decay_rate=0.05, created_at=0
        ))
        ws.decay_annotations(current_tick=10)
        # intensity = 2.0 - 0.05 * 10 = 1.5
        assert ws.annotation_count == 1
        ann = ws.query_annotations(position=(5, 5), radius=0)[0]
        assert abs(ann.intensity - 1.5) < 1e-6

    def test_decay_annotations_uses_delta_since_last_decay(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(
            position=(5, 5), tags={"danger"}, intensity=2.0, decay_rate=0.1, created_at=0
        ))

        ws.decay_annotations(current_tick=10)
        ws.decay_annotations(current_tick=11)

        assert ws.annotation_count == 1
        ann = ws.query_annotations(position=(5, 5), radius=0)[0]
        assert abs(ann.intensity - 0.9) < 1e-6

    def test_decay_annotations_ignores_negative_delta(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(
            position=(5, 5), tags={"danger"}, intensity=2.0, decay_rate=0.1, created_at=0
        ))

        ws.decay_annotations(current_tick=10)
        ws.decay_annotations(current_tick=9)

        ann = ws.query_annotations(position=(5, 5), radius=0)[0]
        assert abs(ann.intensity - 1.0) < 1e-6

    def test_decay_annotations_removes_after_delta_decay(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(
            position=(5, 5), tags={"danger"}, intensity=0.2, decay_rate=0.1, created_at=0
        ))

        ws.decay_annotations(current_tick=2)

        assert ws.annotation_count == 0

    def test_record_event(self):
        ws = BattlefieldWorkspace(map_size=32)
        assert ws.event_count == 0
        ws.record_event(GameEvent(kind="test", agent_id="red_0"))
        assert ws.event_count == 1

    def test_snapshot_annotations_returns_independent_copies(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(
            position=(5, 5),
            tags={"dangerous", "contested"},
            intensity=0.75,
            decay_rate=0.2,
            created_at=3,
            last_decay_tick=4,
        ))

        snapshot = ws.snapshot_annotations()
        snapshot[0].tags.add("mutated")

        original = ws.query_annotations(position=(5, 5), radius=0)[0]
        assert snapshot[0].position == (5, 5)
        assert "mutated" not in original.tags

    def test_export_and_import_annotations_round_trip(self):
        source = BattlefieldWorkspace(map_size=32)
        source.add_annotation(BattlefieldAnnotation(
            position=(4, 7),
            tags={"resource_soon", "objective"},
            intensity=0.6,
            decay_rate=0.03,
            created_at=2,
            last_decay_tick=5,
        ))

        rows = source.export_annotations()
        assert rows == [
            {
                "position": (4, 7),
                "tags": ["objective", "resource_soon"],
                "intensity": 0.6,
                "decay_rate": 0.03,
                "created_at": 2,
                "last_decay_tick": 5,
            }
        ]

        target = BattlefieldWorkspace(map_size=32)
        target.import_annotations(rows)

        imported = target.query_annotations(position=(4, 7), radius=0)[0]
        assert imported.position == (4, 7)
        assert imported.tags == {"objective", "resource_soon"}
        assert imported.intensity == 0.6
        assert imported.decay_rate == 0.03
        assert imported.created_at == 2
        assert imported.last_decay_tick == 5

    def test_observation_layer_shape(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"dangerous"}))
        layer = ws.to_observation_layer(num_channels=3)
        assert layer.shape == (32, 32, 3)

    def test_observation_layer_danger_channel(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"dangerous"}, intensity=0.8))
        layer = ws.to_observation_layer(num_channels=3)
        # channel 0 = danger
        assert layer[5, 5, 0] == 0.8  # note: layer[y, x, c]
        assert layer[0, 0, 0] == 0.0

    def test_observation_layer_opportunity_channel(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(10, 10), tags={"resource_soon"}, intensity=0.6))
        layer = ws.to_observation_layer(num_channels=3)
        assert abs(layer[10, 10, 1] - 0.6) < 1e-6

    def test_observation_layer_control_channel_positive(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(3, 3), tags={"our_control"}, intensity=0.9))
        layer = ws.to_observation_layer(num_channels=3)
        assert abs(layer[3, 3, 2] - 0.9) < 1e-6

    def test_observation_layer_control_channel_negative(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(3, 3), tags={"enemy_control"}, intensity=0.7))
        layer = ws.to_observation_layer(num_channels=3)
        assert abs(layer[3, 3, 2] + 0.7) < 1e-6

    def test_observation_layer_contested_is_negative(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(8, 8), tags={"contested"}, intensity=0.5))
        layer = ws.to_observation_layer(num_channels=3)
        # contested is both danger (ch0) and negative control (ch2)
        assert layer[8, 8, 0] == 0.5
        assert abs(layer[8, 8, 2] + 0.5) < 1e-6

    def test_observation_layer_out_of_bounds_ignored(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(-1, 5), tags={"dangerous"}))
        ws.add_annotation(BattlefieldAnnotation(position=(5, 100), tags={"dangerous"}))
        layer = ws.to_observation_layer(num_channels=3)
        assert np.sum(layer) == 0.0

    def test_clear(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"danger"}))
        ws.record_event(GameEvent(kind="test"))
        ws.clear()
        assert ws.annotation_count == 0
        assert ws.event_count == 0
