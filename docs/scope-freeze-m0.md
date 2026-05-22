# M0 范围冻结记录

生成日期：2026-05-22

## Linear 范围

- 父 Epic：WEN-42【Epic】需求与范围：冻结 HybridArena 双主线维护边界
- M0 子任务：WEN-82、WEN-83、WEN-84、WEN-85
- ISSUE-F13 单独跟踪：WEN-44【Epic】ISSUE-F13：验证 objective policy 与 base objective 可达性
- 后续 Epic：WEN-43（MiniMOBA/RL 稳定化）、WEN-53（AgentBench 完整交付）、WEN-54（发布门禁与维护闭环）、WEN-163（最终回读验收）

## 事实基线

### 关键文档

- `README.md`：项目入口、双主线说明、运行命令与测试命令。
- `docs/plan.md`：当前 RL 主线计划、P0/P1/P2/P3 优先级、AgentBench 维护状态。
- `docs/progress.md`：RL 工程闭环、LLM Planner MVP、AgentBench 首版完成状态。
- `docs/issues.md`：当前阻塞项 ISSUE-F13 与已修复问题。
- `docs/architecture.md`：MiniMOBA/RL 下一阶段架构、模块边界、风险。
- `docs/agentbench-architecture.md`：AgentBench 应用层架构。

### 关键目录

- MiniMOBA/RL：`hybrid_arena/minimoba/`、`hybrid_arena/algorithms/`、`hybrid_arena/training/`、`hybrid_arena/inference/`、`hybrid_arena/scripts/train.py`、`hybrid_arena/scripts/evaluate.py`、`hybrid_arena/scripts/run_ablation.py`。
- AgentBench：`hybrid_arena/core/`、`hybrid_arena/scenarios/`、`hybrid_arena/services/api/`、`hybrid_arena/scripts/agentbench_run.py`、`hybrid_arena/demo/app.py`。
- Demo：`hybrid_arena/demo/moba_app.py`（MOBA 主线）、`hybrid_arena/demo/app.py`（AgentBench）。
- 共享交付：`README.md`、`docs/`、`CHANGELOG.md`、`pyproject.toml`、测试与 lint 门禁。

## 双主线边界

### MiniMOBA/RL 主线

- 职责：PettingZoo Parallel API 环境、324 联合动作与 action mask、ObjectiveSystem、PPO/DualClipPPO/MAPPO/QMIX/COMA、训练/评估/ablation CLI、checkpoint、self-play/curriculum、LLM Planner MVP。
- 当前阶段：RL 工程闭环已完成，正式实验与训练有效性验证为当前重点。
- M1 输入：环境目标闭环、动作系统、算法测试、训练/评估命令、ISSUE-F13。
- 非目标：M0 不实现算法、不改环境核心逻辑、不判断训练效果。

### AgentBench 应用层

- 职责：schema/trace/storage/reporting、JD 解析、通信 RAG、工单分诊、FastAPI、`agentbench_run` CLI、Streamlit AgentBench demo。
- 当前阶段：首版已完成，保留维护；后续功能完整交付归属 M2。
- M2 输入：`core`、`scenarios`、`services/api`、CLI、reporting、demo。
- 非目标：不参与 RL 训练有效性判断，不阻塞 ISSUE-F13 技术验证。

### 共享项

- README、docs、ruff、全量测试、交接记录属于共享维护面。
- Demo/API/docs 风险归属 M3，发布门禁与 Linear 维护闭环归属 M4。
- 任何后续变更必须说明影响哪条主线，并使用对应验收命令验证。

## 冻结测试门禁

```bash
# RL 主线
pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v

# AgentBench 应用层
pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v

# Ruff / lint
ruff check hybrid_arena
```

全量回归在 M4 或交接验收时执行：

```bash
pytest hybrid_arena/ -v
```

### 本轮验证结果

2026-05-22 本地执行结果：

- RL 主线：`96 passed, 1 skipped in 157.59s`
- AgentBench 应用层：`37 passed in 8.97s`
- Ruff / lint：`All checks passed!`

说明：RL 首次合并命令在 124 秒工具超时内未完成；拆分定位后确认 `training/tests` 耗时约 3 分钟，使用更长超时复跑完整 RL 门禁通过。

## 风险与后续输入

- ISSUE-F13 是 MiniMOBA/RL 主线阻塞风险，已由 WEN-44 单独跟踪；M0 只冻结归属与验证入口，不解决该问题。
- M1 必须先验证 scripted objective policy 能否稳定触达 base，再决定是否进入 300k-500k 长训。
- M2 以 AgentBench 的 core/scenarios/API/CLI/reporting 独立验收为边界。
- M3 提前跟踪 Demo/API/docs 口径漂移和命令失效风险。
- M4 负责执行最终发布门禁、Linear 状态回读和跨 repo 交接一致性。
