"""Export the MiniMOBA low-level policy to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hybrid_arena.algorithms.networks import ActorCritic

MODEL_INPUTS = ("local_map", "self_state", "teammate_states", "global_info", "action_mask")
OPSET_VERSION = 17
OBS_CONTRACT = {
    "local_map": [11, 11, 11],
    "self_state": [20],
    "teammate_states": [3, 15],
    "global_info": [10],
    "action_mask": [324],
}
ACTION_CONTRACT = {
    "space": "MultiDiscrete([9, 4, 9])",
    "flat_logits": 324,
    "mask_semantics": "1=legal, 0=illegal",
}


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


def _looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        torch.is_tensor(item) for item in value.values()
    )


def _extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, torch.Tensor], str]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict(), "module"
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a state_dict or mapping with model weights")
    for key in ("model_state_dict", "policy_state_dict", "actor_critic"):
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict(), key
        if _looks_like_state_dict(value):
            return value, key
    if _looks_like_state_dict(checkpoint):
        return checkpoint, "state_dict"
    raise ValueError(
        "unsupported checkpoint format; expected raw state_dict, model_state_dict, "
        "policy_state_dict, or actor_critic"
    )


def load_actor_critic_checkpoint(policy: ActorCritic, checkpoint_path: Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, checkpoint_format = _extract_state_dict(checkpoint)
    load_result = policy.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint_format": checkpoint_format,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def export_policy(
    output: str | Path,
    *,
    seed: int = 7,
    batch_size: int = 1,
    checkpoint: str | Path | None = None,
    export_mode: str = "contract_smoke",
    provider: str = "CPUExecutionProvider",
    device: str = "cpu",
) -> dict:
    if export_mode not in {"contract_smoke", "checkpoint_bound"}:
        raise ValueError("export_mode must be contract_smoke or checkpoint_bound")
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    if checkpoint_path is None and export_mode == "checkpoint_bound":
        raise ValueError("checkpoint_bound export requires --checkpoint")
    if checkpoint_path is not None:
        export_mode = "checkpoint_bound"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    policy = ActorCritic().to(device)
    checkpoint_load: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint_load = load_actor_critic_checkpoint(policy, checkpoint_path)
    model = PolicyONNXWrapper(policy).to(device)
    model.eval()
    dummy_inputs = tuple(
        tensor.to(device) for tensor in make_dummy_inputs(batch_size=batch_size, seed=seed)
    )
    dynamic_axes = {name: {0: "batch"} for name in MODEL_INPUTS}
    dynamic_axes["masked_logits"] = {0: "batch"}
    torch.onnx.export(
        model,
        dummy_inputs,
        output,
        input_names=list(MODEL_INPUTS),
        output_names=["masked_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_sha = model_sha256(output)
    metadata = {
        "output": str(output),
        "seed": seed,
        "batch_size": batch_size,
        "dynamic_axes": "batch",
        "export_mode": export_mode,
        "trained_policy": checkpoint_path is not None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": model_sha256(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_load": checkpoint_load,
        "model_sha256": onnx_sha,
        "sha256": onnx_sha,
        "obs_contract": OBS_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "opset": OPSET_VERSION,
        "provider": provider,
        "device": device,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--export-mode",
        choices=["contract_smoke", "checkpoint_bound"],
        default="contract_smoke",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export_policy(
                args.output,
                seed=args.seed,
                batch_size=args.batch_size,
                checkpoint=args.checkpoint,
                export_mode=args.export_mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
