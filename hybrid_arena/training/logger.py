"""Weights & Biases logging wrapper.

Usage:
    logger = WandbLogger(project="hybrid-arena", config=config_dict)
    logger.log({"reward": 1.2, "win_rate": 0.6}, step=1000)
    logger.finish()
"""

from __future__ import annotations

import importlib.util
from typing import Any


def _has_wandb() -> bool:
    return importlib.util.find_spec("wandb") is not None


class WandbLogger:
    """Lightweight wrapper around wandb.

    Falls back to console print if wandb is not installed.
    """

    def __init__(
        self,
        project: str = "hybrid-arena",
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and _has_wandb()
        self._run = None

        if self.enabled:
            import wandb

            self._run = wandb.init(
                project=project,
                entity=entity,
                group=group,
                name=name,
                config=config,
                reinit=True,
            )
        else:
            print(f"[WandbLogger] wandb disabled or not installed. Group={group}, Name={name}")

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            import wandb

            wandb.log(data, step=step)
        else:
            msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in data.items())
            prefix = f"[Step {step}] " if step is not None else ""
            print(prefix + msg)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            import wandb

            wandb.finish()
            self._run = None

    @property
    def run_id(self) -> str | None:
        if self._run is not None:
            return self._run.id
        return None
