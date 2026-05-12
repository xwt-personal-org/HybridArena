"""Tests for observation helpers: crop_pheromone_layer, append_pheromone_channels."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_arena.minimoba.tactical_runtime.observation import (
    append_pheromone_channels,
    build_augmented_observation,
    crop_pheromone_layer,
)
from hybrid_arena.minimoba.tactical_runtime.workspace import (
    BattlefieldAnnotation,
    BattlefieldWorkspace,
)


class TestCropPheromoneLayer:
    """Tests for crop_pheromone_layer."""

    def test_center_crop_shape(self):
        layer = np.zeros((32, 32, 3), dtype=np.float32)
        crop = crop_pheromone_layer(layer, center=(16, 16), view_size=11)
        assert crop.shape == (11, 11, 3)

    def test_center_crop_values(self):
        layer = np.zeros((32, 32, 3), dtype=np.float32)
        layer[16, 16, 0] = 1.0  # center pixel
        crop = crop_pheromone_layer(layer, center=(16, 16), view_size=11)
        # center of 11x11 is (5, 5)
        assert crop[5, 5, 0] == 1.0

    def test_boundary_crop_zero_pads(self):
        layer = np.zeros((32, 32, 3), dtype=np.float32)
        layer[0, 0, 0] = 1.0  # top-left corner
        crop = crop_pheromone_layer(layer, center=(0, 0), view_size=11)
        assert crop.shape == (11, 11, 3)
        # (0, 0) should be at offset (5, 5) in the crop
        assert crop[5, 5, 0] == 1.0
        # top-left of crop should be zero-padded
        assert crop[0, 0, 0] == 0.0

    def test_single_channel(self):
        layer = np.ones((32, 32, 1), dtype=np.float32)
        crop = crop_pheromone_layer(layer, center=(16, 16), view_size=11)
        assert crop.shape == (11, 11, 1)
        assert np.all(crop == 1.0)

    def test_near_boundary(self):
        layer = np.zeros((32, 32, 2), dtype=np.float32)
        layer[1, 1, 0] = 0.5
        crop = crop_pheromone_layer(layer, center=(1, 1), view_size=11)
        assert crop.shape == (11, 11, 2)
        # center=(1,1), half=5, y_min=-4 → pixel maps to crop[1-(-4)] = crop[5]
        assert abs(crop[5, 5, 0] - 0.5) < 1e-6


class TestAppendPheromoneChannels:
    """Tests for append_pheromone_channels."""

    def test_concat_shape(self):
        local_map = np.zeros((11, 11, 11), dtype=np.float32)
        pheromone = np.zeros((11, 11, 3), dtype=np.float32)
        result = append_pheromone_channels(local_map, pheromone)
        assert result.shape == (11, 11, 14)

    def test_concat_preserves_values(self):
        local_map = np.ones((11, 11, 11), dtype=np.float32)
        pheromone = np.full((11, 11, 3), 0.5, dtype=np.float32)
        result = append_pheromone_channels(local_map, pheromone)
        assert np.all(result[:, :, :11] == 1.0)
        assert np.all(result[:, :, 11:] == 0.5)

    def test_mismatched_spatial_raises(self):
        local_map = np.zeros((11, 11, 11), dtype=np.float32)
        pheromone = np.zeros((13, 13, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Spatial dimensions must match"):
            append_pheromone_channels(local_map, pheromone)


class TestBuildAugmentedObservation:
    """Tests for opt-in augmented observation construction."""

    def test_adds_pheromone_map_without_changing_local_map(self):
        local_map = np.zeros((11, 11, 11), dtype=np.float32)
        observation = {
            "local_map": local_map,
            "self_state": np.zeros((20,), dtype=np.float32),
        }
        workspace = BattlefieldWorkspace(map_size=32)
        workspace.add_annotation(BattlefieldAnnotation(
            position=(16, 16),
            tags={"dangerous"},
            intensity=0.75,
        ))

        augmented = build_augmented_observation(
            observation,
            workspace,
            agent_position=(16, 16),
        )

        assert augmented is not observation
        assert augmented["local_map"] is local_map
        assert "local_map_with_pheromones" not in observation
        assert augmented["local_map"].shape == (11, 11, 11)
        assert augmented["local_map_with_pheromones"].shape == (11, 11, 14)
        assert augmented["local_map_with_pheromones"][5, 5, 11] == 0.75

    def test_raises_when_local_map_missing(self):
        workspace = BattlefieldWorkspace(map_size=32)

        with pytest.raises(KeyError, match="local_map"):
            build_augmented_observation({}, workspace, agent_position=(0, 0))
