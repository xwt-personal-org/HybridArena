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
    make_dummy_inputs,
    model_sha256,
)


def validate_onnx(model: str | Path, *, seed: int = 7, atol: float = 1e-4) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("onnxruntime is required for CPU parity validation") from exc

    torch.manual_seed(seed)
    wrapper = PolicyONNXWrapper(ActorCritic())
    wrapper.eval()
    dummy_inputs = make_dummy_inputs(seed=seed)
    with torch.no_grad():
        torch_logits = wrapper(*dummy_inputs).cpu().numpy()

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    ort_inputs = {name: tensor.cpu().numpy() for name, tensor in zip(MODEL_INPUTS, dummy_inputs, strict=True)}
    ort_logits = session.run(["masked_logits"], ort_inputs)[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    passed = max_abs_diff <= atol
    report = {
        "model": str(model),
        "sha256": model_sha256(model),
        "provider": "CPUExecutionProvider",
        "seed": seed,
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(validate_onnx(args.model, seed=args.seed, atol=args.atol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
