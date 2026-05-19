# RL-MOBA 五阶段路线图

本文只记录阶段边界，不作为机器状态源；机器状态以 `.ai/ledger.json` 为准。

## Phase 1: MARL / Offline RL 基座

- 新增 CTDE global state，供 centralized critic 与 offline learner 使用。
- decentralized execution 不变：actor observation 仍只包含 `local_map`、`self_state`、`teammate_states`、`global_info` 和 `action_mask`。
- 新增 synthetic rule replay schema 与 BC/CQL/IQL CPU smoke 接口。
- synthetic replay 只代表确定性脚本策略，不代表真人 replay。
- objective reachability guard 禁止在 `hard_win_rate`、`base_exposed_rate`、`avg_base_damage`、`avg_tower_damage` 全为 0 时宣称训练有效。

## Phase 2: 分布式训练骨架

默认目标是本地 Actor/Learner 逻辑解耦，不依赖真实集群。

## Phase 3: 推理部署路径

默认目标是 PyTorch 到 ONNX 导出与 ONNX Runtime CPU parity。TensorRT 和 C++ ONNX Runtime 属于外部环境能力。

## Phase 4: LLM Planner x RL Controller

默认仍使用 deterministic `RulePlanner`。LLM 输出必须经过 schema 校验与 action legality adapter。

## Phase 5: QA / Rating / Tournament

默认 rating 为 Elo，回归门禁必须包含 objective metrics，不能用 reward-only improvement 替代。
