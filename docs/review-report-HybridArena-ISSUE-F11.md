# 代码审查报告：HybridArena ISSUE-F11 / sanity_2v2 重训闭环

## 元信息

- 审查日期：2026-05-06
- 审查范围：
  - 最新文档提交：`3820a0e0daf792389f49dd155376d344a3123a13`
  - ISSUE-F11 代码修复提交：`a5a626baa50b0b8c8cd47700dc9159c82b8b6b34`
  - 核心文件：
    - `hybrid_arena/training/evaluator.py`
    - `hybrid_arena/minimoba/env.py`
    - `hybrid_arena/minimoba/game_engine.py`
    - `hybrid_arena/training/tests/test_evaluator_metrics.py`
    - `hybrid_arena/minimoba/tests/test_reward.py`
    - `docs/report.md`
- 对照目标：
  - evaluator 不再跨 red/blue 两队平均 reward
  - draw 不误发 win/lose terminal reward
  - sanity_2v2 重训验证是否能作为下一阶段长训依据

## 总体结论

ISSUE-F11 的核心 evaluator 统计修复方向正确：`avg_reward` 已改为 red team reward，并新增 `avg_red_reward`、`avg_blue_reward`、`avg_reward_margin`。`env.py` 对严格 draw 的 win/lose reward 误发也已修正。

但当前训练闭环仍不应进入 300k-500k 长训。`avg_len=200` 暴露的不是单纯“reward 权重不足”，而是**终局语义和评估指标没有区分 base-destroy win 与 timeout adjudicated win**。当前 `get_winner()` 在 timeout 时会按塔/金币/击杀优势返回 red/blue，`env.py` 会继续发 win/lose reward。这会强化“拖到 timeout 取得优势”，而不是强化“推基地终结”。

最新 GitHub 状态显示本次文档结果已提交并推送，不需要重复提交；但建议新增一个小型 fix/patch 任务修正终局语义与指标口径。

---

## 问题清单

### 🟠 H-1：timeout 优势判胜仍会触发 win/lose terminal reward

- 位置：
  - `hybrid_arena/minimoba/env.py::MiniMOBAEnv.step`
  - `hybrid_arena/minimoba/game_engine.py::GameState.get_winner`
- 类型：奖励目标错配 / 评估口径污染
- 现象：
  - `env.py` 在 `game_state.is_game_over()` 后调用 `get_winner()`。
  - `get_winner()` 在 `max_steps` timeout 后，如果 red/blue 在 towers/gold/kills 上领先，会返回 `"red"` 或 `"blue"`，不是 `"draw"`。
  - 因此即使未推掉基地，只要 timeout 时有优势，仍会触发 win/lose terminal reward。
- 影响：
  - agent 可以通过 timeout 优势获得 terminal reward。
  - `win_rate=46.7%` 不能说明策略学会终结。
  - `avg_len=200` 与当前 reward 逻辑共同表明策略被允许、甚至可能被奖励“拖到裁判判胜”。
- 修复建议：
  1. 在 `GameState` 增加 `terminal_reason: Literal["base_destroyed", "timeout", None]`。
  2. base 被摧毁时设置 `terminal_reason="base_destroyed"`。
  3. step_count 达到 max_steps 时设置 `terminal_reason="timeout"`。
  4. `env.py` 只在 `terminal_reason == "base_destroyed"` 时发放 `win/lose` terminal reward。
  5. timeout 胜负只作为评估 adjudication，不作为训练 terminal reward，除非显式配置开启。
- 建议 scope：`review`

### 🟡 M-1：评估指标缺少 hard win / timeout adjudicated win 拆分

- 位置：`hybrid_arena/training/evaluator.py::evaluate_policy`
- 类型：指标设计不足
- 现象：
  - 当前 `win_rate = red_wins / total_games`。
  - 但 red_wins 混合了 base_destroyed win 与 timeout adjudicated win。
- 影响：
  - 50% 阈值不清楚是在衡量真实终结能力，还是 timeout 优势能力。
  - 当前 `avg_len=200` 时，win_rate 对下一阶段长训决策误导性较强。
- 修复建议：
  - 新增：
    - `hard_win_rate`
    - `timeout_win_rate`
    - `timeout_draw_rate`
    - `base_destroyed_count`
    - `timeout_adjudicated_wins`
  - gate 从 `win_rate >= 0.5` 改为：
    - `hard_win_rate >= 0.3` 且 `avg_len < 0.8 * max_steps`，或
    - 至少连续两组 seed 显示 `hard_win_rate` 上升。
- 建议 scope：`review`

### 🟡 M-2：`avg_reward` 当前是 red team 总和，不是 per-agent 平均

- 位置：`hybrid_arena/training/evaluator.py::evaluate_policy`
- 类型：指标可比性问题
- 现象：
  - `episode_red_rewards` 累加所有 red agent reward。
  - `avg_reward == avg_red_reward` 保持兼容，但其语义是 red team total reward。
- 影响：
  - 2v2 与 4v4 不可直接比较。
  - 后续如果 team_size 变化，reward 数值会随 agent 数量变化。
- 修复建议：
  - 保留 `avg_reward` 兼容字段。
  - 新增：
    - `avg_red_team_reward`
    - `avg_blue_team_reward`
    - `avg_red_agent_reward`
    - `avg_blue_agent_reward`
  - 文档明确 `avg_reward` 的 legacy 语义。
- 建议 scope：`auto`

### 🟡 M-3：draw 测试覆盖不足，不能证明 timeout-advantage 不发 terminal reward

- 位置：`hybrid_arena/minimoba/tests/test_reward.py::test_draw_no_win_lose_reward`
- 类型：测试不足
- 现象：
  - 当前测试只构造静止 timeout true draw，并用 reward 总和范围 `(-1, 1)` 粗略判断。
  - 没有覆盖“timeout 时一方有 tower/gold/kill advantage”的场景。
- 影响：
  - H-1 这种 timeout adjudicated win terminal reward 泄漏不会被测出来。
- 修复建议：
  - 新增测试：
    - `test_timeout_advantage_does_not_grant_win_lose_reward_when_terminal_reason_timeout`
    - `test_base_destroyed_grants_win_lose_reward`
    - `test_evaluator_splits_hard_win_and_timeout_win`
- 建议 scope：`auto`

### 🟢 L-1：`docs/report.md` 仍是 `STATUS: NEEDS_REVIEW`

- 位置：`docs/report.md`
- 类型：流程状态
- 现象：
  - GitHub 最新 report 仍标记 `STATUS: NEEDS_REVIEW`。
- 影响：
  - 状态正确；本轮 Review 后不应直接标 READY，因为仍有 H-1/M-1 需要决策。
- 修复建议：
  - 如果接受本报告，保持 `NEEDS_REVIEW` 或改为 `NEEDS_ESCALATION: terminal semantics / objective reward shaping`。
- 建议 scope：`auto`

---

## 计划偏差总览

| 项 | 计划/目标 | 当前实装 | 偏差类型 | 严重度 |
|---|---|---|---|---|
| evaluator reward 口径 | red team 单独统计 | 已实现 | 无 | - |
| draw 不发 win/lose | 严格 draw 已修 | timeout 优势判胜仍发 terminal reward | 目标错配 | High |
| sanity_2v2 gate | win_rate ≥ 50% 后上长训 | 46.7%，avg_len=200 | 未达 gate | High |
| 终结能力 | 学会推基地 | 未学会，靠 timeout 优势 | 指标缺失 | High |
| 测试覆盖 | 覆盖 terminal reward | 未覆盖 timeout advantage | 测试不足 | Medium |

## 执行端已知问题对照

| report.md 已记录 | 审查发现 | 状态 |
|---|---|---|
| evaluator 修复有效 | 与代码一致 | 已确认 |
| avg_len=200，未学会终结 | 与代码/指标逻辑一致 | 已确认 |
| 需 objective/reward shaping | 同意，但要先拆终局语义 | 需补计划 |
| 不建议直接长训 | 同意 | 保持阻塞 |

## 正面发现

- `evaluator.py` 已停止跨 red/blue 平均 reward。
- `avg_red_reward`、`avg_blue_reward`、`avg_reward_margin` 对诊断有价值。
- 文档把 `avg_len=200` 明确记录为核心问题，这是正确判断。
- 最新文档结果已进入 GitHub 提交，不存在“还没 commit/push”的仓库端事实。

## 是否建议修复

建议修复，但不是重训参数微调，而是先做一个小型 fix/patch：

1. 拆分终局原因：base_destroyed vs timeout。
2. 拆分评估指标：hard_win_rate vs timeout_win_rate。
3. 调整 terminal win/lose reward：只奖励真实终结，timeout adjudication 不发或单独配置。
4. 增加针对 timeout advantage 的单测。
5. 重跑 sanity_2v2 100k，不上 300k-500k。

## 建议执行端修复范围

优先修：

- H-1：timeout 优势判胜 terminal reward 泄漏 / 目标错配
- M-1：hard win 与 timeout adjudicated win 指标拆分
- M-3：新增测试覆盖

暂缓修：

- M-2：per-agent reward 归一化，可与下一轮指标清理一起做
