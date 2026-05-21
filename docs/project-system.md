# HybridArena GitHub Project System

本文件定义 HybridArena 的 GitHub 项目管理体系。目标是把长期研究/工程任务拆成可追踪、可验收、可复盘的 Issue / PR / Project 工作流，避免阶段推进后丢失原始设计和当前进度。

## 1. 项目对象模型

```text
GitHub Project: HybridArena Research & Engineering
└── Epic Issue: 长期目标、研究主线或跨模块工程任务
    ├── Phase Issue: 可验收阶段，通常 1-5 天或一个明确实验批次
    ├── Experiment Issue: 训练、评估、seed sweep、ablation 等实验任务
    ├── Bug / Blocker Issue: 阻塞主线的问题
    └── PR: 具体代码变更，必须链接回对应 Issue
```

### Issue 类型

| 类型 | 用途 | 关闭条件 |
|---|---|---|
| `[Epic]` | 长周期主线，承载原始目标、边界、成功标准和阶段路线 | 所有核心阶段完成并有复盘 |
| `[Phase]` | Epic 下的阶段任务 | 有实现/实验证据，验收标准达成 |
| `[Experiment]` | 训练、评估、ablation、seed sweep | 产出结果文件、指标摘要和结论 |
| `[Bug]` | 明确缺陷 | 复现路径消失，回归测试通过 |
| `[Blocker]` | 阻塞主线的问题 | 有替代方案或阻塞被消除 |
| `[Maintenance]` | 非主线维护 | 维护动作完成并验证 |

## 2. GitHub Project 字段设计

建议创建 GitHub Project：`HybridArena Research & Engineering`。

字段：

| 字段 | 类型 | 建议值 |
|---|---|---|
| Status | Single select | Inbox, Design, Ready, In Progress, Blocked, Review, Done, Archived |
| Type | Single select | Epic, Phase, Experiment, Bug, Blocker, Maintenance |
| Area | Single select | MiniMOBA, RL Training, Algorithms, Planner, AgentBench, CI, Docs |
| Priority | Single select | P0, P1, P2, P3 |
| Stage | Single select | Planning, Build, Test, Experiment, Analysis, Launch, Follow-up |
| Owner | Person | 当前负责人 |
| Target date | Date | 目标完成日期 |
| Last update | Date | 最近一次有效进展更新 |
| Risk | Single select | Low, Medium, High |
| Confidence | Single select | High, Medium, Low |
| Evidence | Text | PR / result file / report 链接 |

## 3. 推荐视图

### Active Epics

过滤：

```text
Type = Epic AND Status != Done AND Status != Archived
```

用途：只看当前长期主线。

### Current Sprint

过滤：

```text
Status IN (Ready, In Progress, Review, Blocked)
```

按 `Priority` 和 `Stage` 排序。

### Blocked / Risk

过滤：

```text
Status = Blocked OR Risk = High
```

用途：每次工作前优先处理阻塞。

### Experiments

过滤：

```text
Type = Experiment
```

按 `Stage` 分组：Planning / Experiment / Analysis / Done。

### Stale Work

过滤建议：

```text
Status != Done AND Last update older than 7 days
```

如果 Project 无法自动判断日期，每周手动检查。

### Roadmap

使用 `Target date` 做时间线视图：

- P0：代码真实性、CI 加固、ISSUE-F13
- P1：正式 baseline、ablation、训练有效性判定
- P2：Self-play / Curriculum 调优
- P3：LLM Planner 正式训练

## 4. 任务推进规则

1. 所有复杂任务必须先建 `[Epic]`。
2. Epic 必须写清楚背景、目标、非目标、成功标准和阶段拆解。
3. 每个阶段必须拆成 `[Phase]` 或 `[Experiment]` Issue。
4. 每个 PR 必须写 `Part of #issue` 或 `Closes #issue`。
5. 任何方向变化必须在 Issue 评论里记录 `Decision Update`。
6. 关闭 Issue 前必须填写验证证据：测试命令、实验输出、报告文件或 PR。
7. Project 只做状态视图，Issue 是唯一事实源。

## 5. 每周复盘流程

每周至少一次执行：

```text
1. 打开 Active Epics
2. 检查 Blocked / Risk
3. 检查 Stale Work
4. 更新每个 Epic 的 Current Status
5. 将完成项移到 Done
6. 将废弃项关闭为 not planned，并写明原因
7. 把下一周最多 3 个核心 Issue 标为 Ready / In Progress
```

## 6. HybridArena 当前主线映射

### P0：代码真实性与正式实验准备

- 本地完整验证矩阵
- CI 覆盖范围扩展
- 文档口径统一
- ISSUE-F13：objective reward shaping 未产生 hard win

### P1：正式 RL Baseline 实验

- `configs/experiments/phase_b_baseline.yaml`
- `run_ablation` 配置文件驱动
- 正式 baseline 训练与评估
- 训练有效性判定脚本
- `win_rate` 与 objective event 统计审计

### P2：Self-play / Curriculum 调优

- 对手池策略
- ELO 门控
- curriculum schedule 调整

### P3：LLM Planner 训练

- planner trace dataset
- GRPO 前置验收
- QLoRA GRPO 正式训练

## 7. 状态更新模板

Issue 评论建议使用：

```md
## Status Update - YYYY-MM-DD

Current phase:
-

Completed:
-

Next:
-

Blocked:
-

Decision changes:
-

Evidence:
-
```

## 8. 实验证据最低要求

实验 Issue 关闭前至少包含：

- 命令行参数或配置文件路径
- seed 列表
- checkpoint / result 路径
- 核心指标：win_rate, hard_win_rate, tower_damage, base_exposed_rate, avg_base_damage, FPS
- 结论：继续、回滚、调参、扩大训练或终止

## 9. PR 验收最低要求

PR 必须说明：

- 关联 Issue
- 变更范围
- 验证命令
- 风险等级
- 是否影响训练结果口径
- 是否需要更新 README / docs / changelog
