# HybridArena 开发进度

## 当前状态

- 当前主线：MiniMOBA / RL + LLM Planner 混合架构
- 最后更新：2026-05-18
- 状态：RL 工程闭环已搭好，进入正式实验与训练有效性验证阶段

## RL 主线进度

### 环境与基础设施（已完成）

- [x] MiniMOBA PettingZoo Parallel API 环境
- [x] MultiDiscrete([9,4,9]) 324 联合动作 + action mask
- [x] 战争迷雾、英雄系统、地图生成
- [x] 基准 FPS 测试脚本（目标 >500 FPS）
- [x] PettingZoo API compliance 测试
- [x] 环境正确性测试与 reward 测试

### 训练正确性（已完成）

- [x] 324-way joint categorical policy
- [x] PPO rollout/update 一致性修复（action_masks + old_values）
- [x] Clipped value loss + DualClipPPO dual_clip_fraction 指标
- [x] RolloutBuffer 支持 mask / value 保存
- [x] Trainer 单环境 8 agent 批处理闭环

### 目标系统（已完成框架）

- [x] tower / base StructureState
- [x] ObjectiveSystem + 目标奖励
- [x] 队伍经济更新
- [x] **ISSUE-F13：objective shaping 长训 gate 已通过**（WEN-44 scripted policy hard_win_rate=1.0；允许启动下一阶段长训）

### 算法（已完成框架）

- [x] PPO / DualClipPPO
- [x] MAPPO
- [x] QMIX
- [x] COMA

### 实验系统（已完成）

- [x] train / evaluate / run_ablation CLI
- [x] checkpoint 工具
- [x] Evaluator（胜率、KDA、tower damage、FPS）
- [x] SyncParallelEnvRunner
- [x] Smoke 实验验证流水线可运行

### Self-play / Curriculum（已完成框架）

- [x] SelfPlayPool + ELO
- [x] CurriculumManager

### LLM Planner（已完成 MVP）

- [x] RulePlanner / DummyLLMClient
- [x] LLMPlanner 状态机（analyze → decide → reflect）
- [x] MacroActionAdapter
- [x] PlannerState / StateTranslator
- [x] PlannerTrace schema + JSONL recorder
- [x] play_planner CLI

### 正式实验（待开始）

- [ ] P1.1：正式实验配置
- [ ] P1.2：run_ablation 配置文件驱动
- [ ] P1.3：正式 baseline 训练与评估
- [ ] P1.4：训练有效性判定脚本
- [ ] P1.5：win_rate 视角审计
- [ ] P1.6：objective event 统计

## AgentBench 应用层进度（已完成，保留维护）

- [x] core：schema / trace / SQLite storage / reporting
- [x] scenarios：jd_resume_match / telecom_rag / ticket_triage
- [x] FastAPI 服务层
- [x] agentbench_run CLI
- [x] Streamlit AgentBench demo
- [x] 面试文档（架构、演示脚本、简历条目、benchmark report）

## 验证记录

- `pytest hybrid_arena/ -v`：180 passed, 1 skipped
- `python -m compileall hybrid_arena`：通过
- `ruff check hybrid_arena`：All checks passed
- AgentBench CLI 三场景均已生成 JSON/Markdown report
- WEN-43（2026-05-28）：`pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v`：105 passed, 1 skipped；train/evaluate smoke 通过，输出记录见 `docs/verification-wen-43.md`
- WEN-90（2026-06-02）：ISSUE-F13 长训 gate 判定通过；依据 WEN-44 PR #15 scripted policy 指标 `hard_win_rate=1.0`、`base_exposed_rate=1.0`、`avg_base_damage=2000.0`、`tower_damage=2400.0`，允许启动下一阶段长训。
