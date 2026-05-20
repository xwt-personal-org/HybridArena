from hybrid_arena.deployment.status import detect_deployment_capabilities


def test_missing_cpp_runtime_does_not_verify_cpp_inference():
    status = detect_deployment_capabilities(env={}, which=lambda _name: None)
    assert status["cpp_build_verifiable"] is False
    assert status["cpp_inference_verified"] is False
    assert status["cmake"]["status"] == "missing"
    assert status["onnxruntime_root"]["status"] == "missing"


def test_missing_tensorrt_does_not_verify_tensorrt():
    status = detect_deployment_capabilities(env={}, which=lambda _name: None)
    assert status["tensorrt_verifiable"] is False
    assert status["tensorrt_verified"] is False
    assert status["tensorrt_verification_status"] == "skipped"
