# Agent 架构下一步路线图 - 2026-05-13

## 本 Patch 边界

本次 patch 只落地方案 A 的安全最小切片：

- skill-runtime API 默认禁用 `WRITE_FS`、`RUN_SHELL`、`NETWORK`、`LLM_CALL` 等高风险 effect。
- `/skill-runtime/tools`、`/skill-runtime/policy`、`/skill-runtime/advice` 暴露可解释的策略状态。
- SQLite memory 修正 `created_at` 语义，并提供最小 memory hygiene 诊断。
- MiniMOBA tactical dispatch 对缺失或非法 controller 返回显式失败。

本 patch 不实现真实 LLM 调用、动态工具下载、网络扫描、图数据库、多 Agent orchestrator、队列、分布式状态或子 Agent 生命周期管理。

## 方案 A 优先候选

1. 结构化 memory query 质量指标：补充命中率、过期率、低成功 trace 复用率等离线统计。
2. 工具策略注册表：把 effect、输入输出 schema、默认策略、审计字段集中管理。
3. 主动 onboarding/advisory UX：把安全策略、memory hygiene、工具使用建议转成更稳定的 CLI/API 诊断输出。

## 方案 B/C 后续研究

方案 B/C 仅作为后续研究方向保留：

- 不在当前 patch 引入多 Agent orchestration、任务队列、子 Agent 生命周期管理或分布式协调。
- 不在当前 patch 引入 Neo4j、RDF、知识图谱数据库或其他新外部依赖。
- 后续只有在方案 A 的策略治理、memory 指标和 advisory UX 稳定后，再重新评估方案 B/C 的成本收益。
