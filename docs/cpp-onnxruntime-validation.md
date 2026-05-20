# C++ ONNX Runtime Validation

本文记录 Topic B 的 C++ ONNX Runtime 部署验证结果，不作为机器状态源；机器状态以 `.ai/ledger.json` 为准。

## 本轮环境结论

本轮没有完成 C++ ONNX Runtime 推理验证。原因是当前环境缺少必要外部依赖：

- CMake: missing
- `ONNXRUNTIME_ROOT`: missing
- C++ ONNX Runtime headers/libs: missing
- TensorRT: missing
- CUDA: 仅检测到 `nvidia-smi`，`torch.cuda.is_available()` 为 false；没有 TensorRT engine 或 latency 对比证据

因此：

- `cpp_inference_verified=false`
- `cpp_inference_verification_status=skipped`
- `tensorrt_verified=false`
- `tensorrt_verification_status=skipped`

详细机器可读结果：

- `results/deployment/status.json`
- `results/deployment/cpp_onnxruntime/validation.json`

## 已运行命令

```bash
python -m hybrid_arena.deployment.status --output results/deployment/status.json
python -m json.tool results/deployment/status.json
python -m compileall hybrid_arena/deployment
pytest hybrid_arena/deployment/tests -v
ruff check hybrid_arena/deployment
```

## 因环境缺失而跳过的命令

以下命令只有在 CMake 可用时才能运行：

```bash
cmake -S cpp_inference -B cpp_inference/build
cmake --build cpp_inference/build
cpp_inference/build/observation_codec_test
```

以下目标只有在 `ONNXRUNTIME_ROOT` 指向包含 `include/onnxruntime_cxx_api.h` 与 ONNX Runtime C++ library 的安装目录时才可验证：

```bash
cpp_inference/build/hybrid_arena_infer
```

## 需要的环境变量和依赖

- 安装 CMake，并确保 `cmake` 在 `PATH`。
- 安装 ONNX Runtime C++ 包。
- 设置 `ONNXRUNTIME_ROOT` 到 ONNX Runtime C++ 根目录，该目录应包含：
  - `include/onnxruntime_cxx_api.h`
  - `lib/onnxruntime.lib` 或平台等价库文件
- 如需验证 TensorRT，另需 CUDA/TensorRT/trtexec 可用，并提供 engine build 与 latency 对比记录。

## 未验证项

- C++ observation codec build/test 未运行，因为 CMake 缺失。
- C++ ONNX Runtime demo target 未构建，因为 CMake 与 `ONNXRUNTIME_ROOT` 缺失。
- TensorRT acceleration 未验证。

## 声明边界

本分支只证明 deployment status 能真实报告环境缺口，并把 C++/TensorRT 缺口写入机器可读 validation artifact。它不证明 C++ ONNX Runtime 推理已通过，也不证明 TensorRT 已可用或有加速收益。
