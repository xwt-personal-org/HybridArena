# RL-MOBA 五阶段路线图

本文只记录阶段边界，不作为机器状态源；机器状态以 `.ai/ledger.json` 为准。声明边界的统一表见 `docs/claim-boundaries.md`。

## Phase 1: MARL / Offline RL 基座

- 已有 CTDE global state，可供 centralized critic 和 offline learner 使用。
- decentralized execution 不变：actor observation 仍只包含 `local_map`、`self_state`、`teammate_states`、`global_info` 和 `action_mask`。
- 已有 synthetic rule replay schema 与 BC/CQL/IQL CPU smoke 接口。
- 边界：这是 foundation + CPU smoke，不是 solved offline RL，也不是长训收敛证明。

## Phase 2: 分布式训练骨架

- 默认目标是本地 actor/learner 逻辑解耦，不依赖真实集群。
- 已有 bounded queue、policy version lag 可观测性和 V-trace smoke。
- 边界：这是 local skeleton，不是 cluster throughput proof。

## Phase 3: 推理部署路径

- 默认目标是 PyTorch 到 ONNX 的 contract-smoke 导出与 ONNX Runtime CPU parity。
- 无 checkpoint 时，ONNX artifact 必须标记为 `export_mode=contract_smoke` 和 `trained_policy=false`。
- 只有 checkpoint-bound metadata 和同 checkpoint PyTorch parity 通过时，才能声明 checkpoint-bound policy artifact。
- C++ inference 仍是 skeleton unless CMake + ONNX Runtime C++ build 记录通过。
- TensorRT 未验证，除非 CUDA/TensorRT 命令和 latency 对比记录通过。

## Phase 4: LLM Planner x RL Controller

- 默认仍使用 deterministic `RulePlanner` 或 strict stub provider。
- LLM macro action 必须走 strict validation；未知、空、非字符串或 malformed action 必须 fail closed。
- legacy alias 只允许在 `MacroActionAdapter` / rule 兼容路径中 normalization。
- 边界：当前不是外部真实 LLM gameplay proof。

## Phase 5: QA / Rating / Tournament

- 默认 rating 使用 Elo。
- 默认 tournament 是 rule-smoke QA，报告必须包含 `Evaluation Mode`、`Policy Source`、`Planner Source`、`Claim Boundary` 和 `Open Items`。
- `illegal_action_rate` 与 `planner_override_rate` 必须带 source metadata，不能是无来源常数。
- 边界：默认 QA 不是 current trained policy validation；只有 checkpoint/planner artifact 证明时才能升级口径。
