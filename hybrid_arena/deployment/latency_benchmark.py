"""ONNX Runtime latency benchmark contract."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from hybrid_arena.deployment.export_onnx import MODEL_INPUTS, make_dummy_inputs, model_sha256


@dataclass(frozen=True)
class LatencyReport:
    p50_ms: float
    p95_ms: float
    max_ms: float
    batch_size: int
    provider: str
    device: str
    model_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_onnx(model: str | Path, *, iterations: int = 50, seed: int = 7, batch_size: int = 1) -> LatencyReport:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inputs = make_dummy_inputs(batch_size=batch_size, seed=seed)
    ort_inputs = {name: tensor.cpu().numpy() for name, tensor in zip(MODEL_INPUTS, inputs, strict=True)}
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(["masked_logits"], ort_inputs)
        samples.append((time.perf_counter() - start) * 1000.0)
    return LatencyReport(
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        max_ms=float(np.max(samples)),
        batch_size=batch_size,
        provider="CPUExecutionProvider",
        device="cpu",
        model_hash=model_sha256(model),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    report = benchmark_onnx(
        args.model,
        iterations=args.iterations,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
