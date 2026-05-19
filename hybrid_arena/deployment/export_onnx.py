"""Export the MiniMOBA low-level policy to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from hybrid_arena.algorithms.networks import ActorCritic

MODEL_INPUTS = ("local_map", "self_state", "teammate_states", "global_info", "action_mask")


class PolicyONNXWrapper(nn.Module):
    """ONNX-friendly wrapper that emits masked 324-way logits."""

    def __init__(self, policy: ActorCritic):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        local_map: torch.Tensor,
        self_state: torch.Tensor,
        teammate_states: torch.Tensor,
        global_info: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        obs = {
            "local_map": local_map,
            "self_state": self_state,
            "teammate_states": teammate_states,
            "global_info": global_info,
        }
        features = self.policy.get_features(obs)
        joint_logits = self.policy._build_joint_logits(
            self.policy.move_head(features),
            self.policy.skill_head(features),
            self.policy.target_head(features),
        )
        return joint_logits.masked_fill(action_mask <= 0, -1e8)


def make_dummy_inputs(batch_size: int = 1, seed: int = 7) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand((batch_size, 11, 11, 11), generator=generator),
        torch.rand((batch_size, 20), generator=generator),
        torch.rand((batch_size, 3, 15), generator=generator),
        torch.rand((batch_size, 10), generator=generator),
        torch.ones((batch_size, 324), dtype=torch.float32),
    )


def model_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_policy(output: str | Path, *, seed: int = 7, batch_size: int = 1) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = PolicyONNXWrapper(ActorCritic())
    model.eval()
    dummy_inputs = make_dummy_inputs(batch_size=batch_size, seed=seed)
    dynamic_axes = {name: {0: "batch"} for name in MODEL_INPUTS}
    dynamic_axes["masked_logits"] = {0: "batch"}
    torch.onnx.export(
        model,
        dummy_inputs,
        output,
        input_names=list(MODEL_INPUTS),
        output_names=["masked_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    metadata = {
        "output": str(output),
        "seed": seed,
        "batch_size": batch_size,
        "opset": 17,
        "dynamic_axes": "batch",
        "sha256": model_sha256(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export_policy(args.output, seed=args.seed, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
