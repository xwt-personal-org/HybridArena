# HybridArena 开发进度

## 当前状态

- 当前阶段：全部模块完成
- 最后更新：2026-04-30
- 状态：模块 0-7 已完成，最终验证通过

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

## 已知问题

当前已知问题均已纳入 `docs/plan.md`，Codex 执行时如遇到计划未覆盖的新问题，记录到 `docs/issues.md`。
