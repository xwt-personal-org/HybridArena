# HybridArena 开发进度

## 当前状态

- 当前阶段：Phase F（验证、纠偏与正式实验准备）
- 最后更新：2026-05-03
- 状态：模块 0-7 已完成，Phase F 进行中（F9 DualClipPPO 对照完成）

## 模块进度

### 模块 0：项目基线与开发护栏

- [x] Step 0.1：新增本地开发说明
- [x] Step 0.2：修正 pyproject 的 Python 版本边界
- [x] Step 0.3：整理测试目录
- [x] Step 0.4：新增 smoke test 配置
- [x] Step 0.5：建立 CI 工作流

### 模块 1：动作编码与 Action Mask 语义修复

- [x] Step 1.1：新增动作编码工具
- [x] Step 1.2：把 GameState 中的 magic number 改为动作编码工具
- [x] Step 1.3：新增 action mask 单元测试
- [x] Step 1.4：修复 ActorCritic 的 joint logits
- [x] Step 1.5：新增 joint action policy 测试
- [x] Step 1.6：让 RandomAgent 复用动作编码工具
- [x] Step 1.7：文档同步

### 模块 2：PPO / DualClipPPO 训练闭环修复

- [x] Step 2.1：RolloutBuffer 保存 action_mask
- [x] Step 2.2：RolloutBuffer 返回 old_values
- [x] Step 2.3：Trainer 写入 action_mask
- [x] Step 2.4：PPO.update 接收 old_values
- [x] Step 2.5：修复 clipped value loss
- [x] Step 2.6：Trainer.update 传入 action_masks 和 old_values
- [x] Step 2.7：修复 DualClipPPO 指标

### 模块 3：MiniMOBA Objective Game 补完

- [x] Step 3.1：新增结构物状态模型
- [x] Step 3.2：GameState 初始化 towers/bases
- [x] Step 3.3：实现攻击结构物
- [x] Step 3.4：实现推塔奖励与队伍经济
- [x] Step 3.5：实现基地终局
- [x] Step 3.6：扩展 observation 与 info
- [x] Step 3.7：更新 reward 与 README

### 模块 4：训练 CLI、评估器与实验复现

- [x] Step 4.1：新增 checkpoint 工具
- [x] Step 4.2：Trainer 支持 checkpoint 与 metrics 输出
- [x] Step 4.3：新增 Evaluator
- [x] Step 4.4：新增 train CLI
- [x] Step 4.5：新增 evaluate CLI
- [x] Step 4.6：新增 ablation runner
- [x] Step 4.7：更新 README 的训练与评估章节

### 模块 5：采样效率、Self-play 与 Curriculum

- [x] Step 5.1：实现同步多环境 runner
- [x] Step 5.2：Trainer 使用 `config.num_envs`
- [x] Step 5.3：新增 SelfPlayPool
- [x] Step 5.4：Evaluator 支持 self-play opponent
- [x] Step 5.5：新增 CurriculumManager
- [x] Step 5.6：训练日志输出阶段 B 结果表

### 模块 6：LLM 高层 Planner MVP

- [x] Step 6.1：定义宏观动作集合
- [x] Step 6.2：定义 PlannerState
- [x] Step 6.3：实现 RulePlanner 作为非 LLM 对照
- [x] Step 6.4：实现 LLMPlanner 接口但默认离线可测
- [x] Step 6.5：实现 MacroActionAdapter
- [x] Step 6.6：新增 planner demo 脚本

### 模块 7：项目包装、文档与发布

- [x] Step 7.1：重写 README 的结果章节
- [x] Step 7.2：生成首版实验报告
- [x] Step 7.3：准备 GitHub release v0.2.0

## Phase F：验证、纠偏与正式实验准备

### 模块 F0：建立验证基线

- [x] Step F0.1：安装完整开发依赖
- [x] Step F0.2：运行完整静态验证
- [x] Step F0.3：运行完整测试套件
  - 修复：`env.close()` 中 pygame import 改为 try/except，避免无 pygame 时崩溃

### 模块 F1：扩展 CI，避免只测 minimoba

- [x] Step F1.1：更新 CI 测试范围为 `pytest hybrid_arena -v`
- [x] Step F1.2：CI 安装命令改为 `".[dev,rl]"`，不安装 `llm`

### 模块 F2：修正文档状态口径

- [x] Step F2.1：修正 README GRPO/QLoRA 矛盾
  - 亮点、Phase D 状态、Known Limitations 三者统一为"实验性 / skeleton complete"
- [x] Step F2.2：更新实验报告，增加 smoke 参数说明、正式 benchmark 警告、下一步实验矩阵

### 模块 F3：正式 Baseline 实验矩阵

- [x] Step F3.1：创建 `configs/experiments/baseline_v1.yaml`
- [x] Step F3.2：`run_ablation.py` 支持 `--config` 和 `--dry-run`
- [x] Step F3.3：创建并验证 `configs/experiments/baseline_smoke.yaml`

### 模块 F4：审计 win_rate 与 objective 指标

- [x] Step F4.1：新增 evaluator 指标一致性测试，包含 episodes 字段
- [x] Step F4.2：结果表增加 `draw_rate`、`avg_towers_destroyed`、`avg_tower_hp_advantage`

### 模块 F5：Planner Trace 数据接口

- [x] Step F5.1：定义 `PlannerTrace` dataclass + schema
- [x] Step F5.2：实现 `PlannerTraceRecorder`（JSONL 输出）
- [x] Step F5.3：`play_planner.py` 支持 `--trace-output` 参数

### 模块 F6：更新进度与问题文档

- [x] Step F6.1：更新 `docs/progress.md`
- [x] Step F6.2：更新 `docs/issues.md`
- [x] Step F6.3：更新 `CHANGELOG.md`

### 模块 F7：正式 Baseline 实验

- [x] Step F7.1：修改 `run_ablation.py` 支持训练+评估流程
- [x] Step F7.2：运行 baseline_v1.yaml 正式实验（ppo seed=42, 100k steps）
- [x] Step F7.3：生成 `results/baseline_v1/ablation_raw.csv` 与 `ablation_summary.md`
- [x] Step F7.4：更新 README 实验结果表
- [x] Step F7.5：运行 planner trace smoke 测试，确认 JSONL 可生成

### 模块 F8：同步收口 baseline_v1 partial 结果

- [x] Step F8.1：提交 `run_ablation.py` 的 train_eval / eval_only / baseline 模式改动
- [x] Step F8.2：复制 `results/baseline_v1/` 结果到 `docs/experiment-report-v0.md`
- [x] Step F8.3：更新 README 实验表标注为 `baseline_v1 partial: ppo seed=42 100k steps`
- [x] Step F8.4：在 `docs/issues.md` 记录训练信号较弱（ISSUE-F9）
- [x] Step F8.5：新增下一轮建议：优先跑 `ppo_dualclip seed=42` 同配置对照

### 模块 F9：DualClipPPO 对照实验

- [x] Step F9.1：训练 ppo_dualclip seed=42（100k steps，CPU ~72 分钟）
- [x] Step F9.2：评估 ppo_dualclip seed=42 vs random（30 episodes）→ win_rate=0.0%
- [x] Step F9.3：评估 ppo_dualclip seed=42 vs rule_based（30 episodes）→ win_rate=0.0%
- [x] Step F9.4：更新 baseline_v1 partial 结果表
- [x] Step F9.5：更新 experiment-report-v0.md、README.md、progress.md
- [x] Step F9.6：DualClipPPO 未改善（对 random 0% vs ppo 16.7%），记录为训练信号诊断入口，不继续完整矩阵

### 模块 F10：训练信号诊断 — sanity_2v2

- [x] Step F10.1：创建 `configs/experiments/sanity_2v2.yaml`（2v2, map_size=16, max_steps=200）
- [x] Step F10.2：修改 Trainer 支持 training_curve 累积和 CSV 导出
- [x] Step F10.3：训练 PPO seed=42, 100k steps（2v2 simplified, 2069s）
- [x] Step F10.4：评估 vs random, 30 episodes → win_rate=33.3%, draw_rate=33.3%
- [x] Step F10.5：导出训练曲线 CSV（781 数据点）
- [x] Step F10.6：对比 4v4 baseline，确认环境复杂度是因素之一
- [x] Step F10.7：诊断 evaluator 跨两队平均问题 + draw_rate timeout 问题
- [x] Step F10.8：记录 ISSUE-F11，更新 experiment-report-v0.md、issues.md、report.md

## 已知问题

当前已知问题均已纳入 `docs/plan.md`，Codex 执行时如遇到计划未覆盖的新问题，记录到 `docs/issues.md`。
