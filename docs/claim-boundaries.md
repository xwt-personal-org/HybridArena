# Claim Boundaries

本文只记录人工可读的声明边界；机器状态以 `.ai/ledger.json` 为准。

| 功能面 | 当前证据 | 不能声明 | 升级声明需要的证据 |
|---|---|---|---|
| MARL / Offline RL | CTDE state、synthetic replay schema、BC/CQL/IQL CPU smoke | 已解决 offline RL 或已完成稳定训练闭环 | 真实 replay、长训曲线、多 seed 指标、objective reachability 通过 |
| Distributed training | 本地 actor/learner skeleton、bounded queue、V-trace smoke | 集群吞吐、容错或生产级分布式训练 | 多进程/多机运行记录、吞吐和版本滞后审计 |
| ONNX | 无 checkpoint 时仅 `contract_smoke` export 和 CPU parity | trained policy artifact | metadata 中 `export_mode=checkpoint_bound`、`trained_policy=true`、checkpoint hash 与 parity report |
| C++ inference | C++ skeleton / contract only | C++ ONNX Runtime 推理已验证 | CMake configure/build 成功，`ONNXRUNTIME_ROOT` headers/libs 可用，运行记录写入报告 |
| TensorRT | 当前未验证 | TensorRT 可用或有加速收益 | CUDA/TensorRT/trtexec 命令记录、engine build、latency 对比 |
| LLM planner | strict stub/rule validation、macro adapter smoke | 外部真实 LLM gameplay proof | 外部 provider 配置、prompt/output trace、action legality audit、多局评估 |
| QA tournament | 默认 rule-smoke tournament，报告写明 `policy_source` 和 `evaluation_mode` | current trained policy validation | checkpoint/planner source、artifact hash、metric source 与 claim boundary 一并记录 |

当前默认口径：所有 smoke 结果只能证明接口、合约和基本回归门没有回退，不能升级为训练完成、生产部署或真实 LLM 对局证明。
