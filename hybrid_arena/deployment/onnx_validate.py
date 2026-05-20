"""Validate ONNX Runtime CPU parity against the PyTorch policy wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.deployment.export_onnx import (
    MODEL_INPUTS,
    PolicyONNXWrapper,
    load_actor_critic_checkpoint,
    make_dummy_inputs,
    model_sha256,
)


def _metadata_path(model: Path) -> Path:
    return model.with_suffix(model.suffix + ".json")


def _load_metadata(model: Path) -> dict:
    path = _metadata_path(model)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_onnx(
    model: str | Path,
    *,
    seed: int = 7,
    atol: float = 1e-4,
    checkpoint: str | Path | None = None,
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("onnxruntime is required for CPU parity validation") from exc

    model_path = Path(model)
    metadata = _load_metadata(model_path)
    checkpoint_path = checkpoint or metadata.get("checkpoint_path")
    if metadata.get("trained_policy") is True and not checkpoint_path:
        raise ValueError("trained_policy metadata requires checkpoint_path for parity")

    torch.manual_seed(seed)
    policy = ActorCritic()
    checkpoint_load = None
    if checkpoint_path:
        checkpoint_load = load_actor_critic_checkpoint(policy, Path(checkpoint_path))
    wrapper = PolicyONNXWrapper(policy)
    wrapper.eval()
    dummy_inputs = make_dummy_inputs(seed=seed)
    with torch.no_grad():
        torch_logits = wrapper(*dummy_inputs).cpu().numpy()

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: tensor.cpu().numpy() for name, tensor in zip(MODEL_INPUTS, dummy_inputs, strict=True)
    }
    ort_logits = session.run(["masked_logits"], ort_inputs)[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    passed = max_abs_diff <= atol
    report = {
        "model": str(model_path),
        "sha256": model_sha256(model_path),
        "provider": "CPUExecutionProvider",
        "seed": seed,
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "passed": passed,
        "export_mode": metadata.get("export_mode", "unknown"),
        "trained_policy": bool(metadata.get("trained_policy", False)),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "checkpoint_load": checkpoint_load,
    }
    if not passed:
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_onnx(args.model, seed=args.seed, atol=args.atol, checkpoint=args.checkpoint),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
