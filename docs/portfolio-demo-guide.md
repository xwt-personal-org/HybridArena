# Portfolio Demo Guide

本文只记录 HybridArena 当前可展示的作品集口径，不作为机器状态源；机器状态以 `.ai/ledger.json` 为准。

## 项目定位

HybridArena 是一个 MiniMOBA 强化学习工程样例，重点展示 RL 环境、训练接口、ONNX 导出、C++ 推理骨架、LLM Planner 安全边界和 QA tournament 闭环如何串起来。当前稳定展示分支的定位是“可复现工程闭环与证据边界”，不是生产级 MOBA AI，也不是已经完成高水平训练的策略。

## 可安全演示内容

- MiniMOBA 环境、动作空间和 action mask 合约。
- Rule-based / smoke policy 的 QA tournament 报告生成。
- LLM macro action 的 fail-closed 行为：未知、空值、非字符串或 malformed action 不会绕过 action mask。
- ONNX `contract_smoke` 导出与 CPU parity；没有 checkpoint 时不能声称 trained policy artifact。
- deployment status 能明确报告 CMake、ONNX Runtime C++、CUDA/TensorRT 的环境缺口。
- claim-boundary 文档和测试用于约束口径，避免把 smoke 结果升级为训练质量、C++ 部署或 TensorRT 加速证明。

## QA Smoke 命令

```bash
python -m hybrid_arena.scripts.qa_tournament --episodes 2 --seed 7 --output results/qa/portfolio_smoke
```

期望产物：

- `results/qa/portfolio_smoke/qa_tournament.json`
- `results/qa/portfolio_smoke/qa_tournament.csv`
- `results/qa/portfolio_smoke/qa_tournament.md`

报告中应包含 `Evaluation Mode`、`Policy Source`、`Planner Source`、`Claim Boundary` 和 `Open Items`。默认 smoke 口径是 rule / macro adapter 证据，不是 current trained policy validation。

## Deployment Status 命令

```bash
python -m hybrid_arena.deployment.status --output results/deployment/status.json
python -m json.tool results/deployment/status.json
```

当前展示分支只应按 status 结果说明能力：

- `python_onnx_export` 和 `python_onnxruntime` 可用于 Python 侧 ONNX smoke / parity。
- `cpp_inference_verified=false` 时，不能声称 C++ ONNX Runtime 推理已验证。
- `tensorrt_verified=false` 时，不能声称 TensorRT 已可用或已有加速收益。

## 已知 Open Items

- 当前环境未完成 `cpp_inference` 的 CMake configure/build 证据。
- `ONNXRUNTIME_ROOT` 未验证可用时，C++ ONNX Runtime headers/libs 仍是缺口。
- CUDA/TensorRT 加速未验证；没有 TensorRT engine build 或 latency 对比证据。
- 真实 checkpoint 证据链在 Topic A 分支深化；稳定展示分支只保留 claim-boundary baseline。
- C++ ONNX Runtime 部署验证在 Topic B 分支深化；稳定展示分支不合入未审查结果。

## 面试讲解口径

- 可以说：项目已经有 MiniMOBA RL 工程闭环、QA smoke、claim boundary、ONNX Python 侧导出/验证和部署能力探测。
- 可以说：LLM Planner 当前采用严格 schema 和 fail-closed 策略，避免模型输出直接驱动低层动作。
- 可以说：ONNX metadata 明确区分 `contract_smoke` 和 `checkpoint_bound`，防止把随机初始化 smoke artifact 误称为训练策略。
- 不应说：已经训练出高水平 MOBA 策略。
- 不应说：C++ ONNX Runtime 或 TensorRT 已完成部署验证，除非 Topic B 分支提供实际命令证据。
- 不应说：外部真实 LLM 已完成 gameplay proof，当前只有 stub / rule / macro adapter 的受控路径。
