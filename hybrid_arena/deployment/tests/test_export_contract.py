import torch

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.deployment.export_onnx import (
    MODEL_INPUTS,
    export_policy,
    load_actor_critic_checkpoint,
    make_dummy_inputs,
)
from hybrid_arena.deployment.latency_benchmark import LatencyReport
from hybrid_arena.deployment.onnx_validate import validate_onnx


def test_dummy_inputs_match_export_contract():
    inputs = make_dummy_inputs(batch_size=2, seed=7)
    assert len(inputs) == len(MODEL_INPUTS)
    assert inputs[0].shape == (2, 11, 11, 11)
    assert inputs[4].shape == (2, 324)


def test_export_and_validate_onnx_cpu_parity(tmp_path):
    output = tmp_path / "policy.onnx"
    metadata = export_policy(output, seed=7)
    report = validate_onnx(output, seed=7)
    assert output.exists()
    assert metadata["sha256"] == report["sha256"]
    assert metadata["export_mode"] == "contract_smoke"
    assert metadata["trained_policy"] is False
    assert metadata["checkpoint_path"] is None
    assert report["passed"] is True


def test_checkpoint_bound_export_metadata_and_parity(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    torch.save({"model_state_dict": ActorCritic().state_dict()}, checkpoint)
    output = tmp_path / "policy-checkpoint.onnx"

    metadata = export_policy(output, seed=7, checkpoint=checkpoint)
    report = validate_onnx(output, seed=7)

    assert metadata["export_mode"] == "checkpoint_bound"
    assert metadata["trained_policy"] is True
    assert metadata["checkpoint_path"] == str(checkpoint)
    assert metadata["checkpoint_sha256"]
    assert metadata["model_sha256"] == report["sha256"]
    assert report["passed"] is True
    assert report["trained_policy"] is True


def test_checkpoint_loader_supports_raw_state_dict(tmp_path):
    checkpoint = tmp_path / "raw.pt"
    torch.save(ActorCritic().state_dict(), checkpoint)
    load_report = load_actor_critic_checkpoint(ActorCritic(), checkpoint)
    assert load_report["checkpoint_format"] == "state_dict"


def test_latency_report_contract():
    report = LatencyReport(
        p50_ms=1.0,
        p95_ms=2.0,
        max_ms=3.0,
        batch_size=1,
        provider="CPUExecutionProvider",
        device="cpu",
        model_hash="abc",
    )
    assert set(report.to_dict()) == {
        "p50_ms",
        "p95_ms",
        "max_ms",
        "batch_size",
        "provider",
        "device",
        "model_hash",
    }
