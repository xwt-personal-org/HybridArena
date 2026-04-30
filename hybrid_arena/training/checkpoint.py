"""Checkpoint save/load helpers for HybridArena training."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _config_to_dict(config: Any) -> dict:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return dict(getattr(config, "__dict__", {}))


def save_checkpoint(
    path: str | Path,
    network,
    optimizer,
    config,
    global_step: int,
    metrics: dict | None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": _config_to_dict(config),
            "global_step": global_step,
            "metrics": metrics or {},
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    network=None,
    optimizer=None,
    map_location: str = "cpu",
) -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model_state = checkpoint.get("model_state_dict") or checkpoint.get("network_state_dict")
    if network is not None and model_state is not None:
        network.load_state_dict(model_state)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return checkpoint
