import torch

from hybrid_arena.algorithms.networks import ActorCritic
from hybrid_arena.deployment.export_onnx import export_policy
from hybrid_arena.deployment.status import detect_deployment_capabilities


def test_contract_smoke_export_is_not_trained_policy(tmp_path):
    metadata = export_policy(tmp_path / "contract-smoke.onnx", seed=7)
    assert metadata["export_mode"] == "contract_smoke"
    assert metadata["trained_policy"] is False
    assert metadata["checkpoint_path"] is None


def test_fixture_checkpoint_export_is_checkpoint_bound(tmp_path):
    checkpoint = tmp_path / "fixture-policy.pt"
    torch.save({"model_state_dict": ActorCritic().state_dict()}, checkpoint)
    metadata = export_policy(tmp_path / "checkpoint-bound.onnx", seed=7, checkpoint=checkpoint)
    assert metadata["export_mode"] == "checkpoint_bound"
    assert metadata["trained_policy"] is True
    assert metadata["checkpoint_sha256"]


def test_missing_external_builds_do_not_claim_verified():
    status = detect_deployment_capabilities(env={}, which=lambda _name: None)
    assert status["cpp_inference_verified"] is False
    assert status["tensorrt_verified"] is False
    assert status["cpp_build_verifiable"] is False
    assert status["tensorrt_verifiable"] is False
