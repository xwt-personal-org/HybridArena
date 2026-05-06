# HybridArena 开发计划 v2：Objective Reward Shaping

## 元信息

- 项目：HybridArena
- 版本：v2
- 创建日期：2026-05-06
- 最后更新：2026-05-06
- 技术栈：Python 3.10+、PettingZoo、Gymnasium、NumPy、PyTorch、Pytest、Ruff
- 基于版本：`docs/plan.md` v1 + `docs/report.md` ISSUE-F12 重训结果
- 变更类型：patch / 架构级训练目标修正
- 总模块数：原模块 0-6 保留不变；新增 Phase F13
- 本版目标：通过 objective reward shaping 让 sanity_2v2 出现真实基地摧毁胜利，即 `hard_win_rate > 0`

## 变更记录

| 版本 | 日期 | 变更摘要 |
|---|---|---|
| v1 | 2026-04-29 | 原始 HybridArena 下一步完善开发计划 |
| v2 | 2026-05-06 | 新增 Phase F13：objective reward shaping，解决 `hard_win_rate=0` 与 timeout-only win 问题 |

## Status

> 任何 agent 读到此区块即可恢复本轮变更上下文。

- 当前阶段：Phase F13 Step F13.1
- 整体进度：原模块 0-6 与 Phase F0-F12 按当前 `docs/progress.md` 保留；本轮新增 9 个步骤
- 状态：变更后待执行
- 阻塞项：
  - `hard_win_rate=0.000`
  - `timeout_win_rate=0.400`
  - `avg_len=200.0`
  - 300k-500k 长训继续阻塞
- Last Iteration Summary：
  - ISSUE-F12 已拆分 terminal semantics。
  - timeout adjudicated win 不再发 `win/lose` terminal reward。
  - 重训 sanity_2v2 后 `win_rate=0.400`，但 `hard_win_rate=0.000`，所有胜利仍来自 timeout adjudication。
- Pending Decisions：
  - 无需外部选型。
  - 本轮不调大 `win/lose` terminal reward。
  - 本轮不恢复 timeout terminal reward。
  - 若 F13 重训后 `hard_win_rate` 仍为 0，进入下一轮 Iter：macro-action curriculum / path-to-objective shaping。

## 不变范围

以下内容保持当前 `docs/plan.md` v1 原样，不在本轮重写：

- 模块 0：项目基线与开发护栏
- 模块 1：动作编码与 Action Mask 语义修复
- 模块 2：PPO / DualClipPPO 训练闭环修复
- 模块 3：MiniMOBA Objective Game 补完
- 模块 4：训练 CLI、评估器与实验复现
- 模块 5：采样效率、Self-play 与 Curriculum
- 模块 6：LLM 高层 Planner MVP
- Phase F0-F12：按 `docs/progress.md` 的已完成状态保留

本轮只新增 Phase F13，并允许执行端按 `docs/inbox/plan.md` merge-back 到当前计划。


# Phase F13：Objective Reward Shaping for Hard Wins

## 概述

- 职责：引入可控的 objective-progress dense reward，让 PPO 在 2v2 sanity 环境中学会从推塔过渡到推基地，而不是只在 timeout 时靠 objective advantage 判胜。
- 前置依赖：
  - ISSUE-F11 evaluator reward 口径已修。
  - ISSUE-F12 terminal semantics 已修。
  - `run_ablation.py` 已导出 `hard_win_rate` / `timeout_win_rate` / `timeout_draw_rate`。
- 预计步骤数：9
- 设计约束：
  - 不放大 `win` / `lose`。
  - timeout win 继续不发 terminal reward。
  - shaping reward 必须可配置、可关闭、可诊断。
  - reward 尺度必须低于 kill/win 主奖励，避免 reward hacking。
  - 先在 sanity_2v2 100k 验证，不上 300k-500k。

## Step F13.1：扩展 `RewardConfig` 的 objective shaping 字段

- **scope: auto**
- 操作：
  - 修改 `hybrid_arena/minimoba/reward_shaper.py`。
  - 在 `RewardConfig` 增加字段：
    - `objective_tower_damage_team: float = 0.001`
    - `objective_base_damage_team: float = 0.003`
    - `objective_base_exposed_team: float = 1.0`
    - `objective_step_cap_team: float = 0.25`
    - `objective_enabled: bool = True`
- 语义：
  - tower/base damage 按实际伤害给全队 shared reward。
  - base damage 权重大于 tower damage。
  - enemy base 首次暴露时给一次性 team reward。
  - 单 step objective shared reward 受 cap 限制。
  - `objective_enabled=False` 时行为等价于 F12。
- 验证：
  - 新增或更新 `hybrid_arena/minimoba/tests/test_reward_config.py`
  - 测试：`test_reward_config_objective_defaults`、`test_reward_config_objective_can_disable`
  - 命令：`pytest hybrid_arena/minimoba/tests/test_reward_config.py -v`

## Step F13.2：实现 team-shared objective reward 分配 helper

- **scope: review**
- 操作：
  - 修改 `hybrid_arena/minimoba/game_engine.py`。
  - 新增私有方法：
    - `_add_team_reward(step_rewards: dict[str, float], team: str, amount: float, *, divide_by_team_size: bool = True) -> None`
- 规则：
  - `team` 只能是 `"red"` 或 `"blue"`。
  - 默认按 `team_size` 平分到该队所有 `possible_agents`，不只给 alive agent。
  - `amount <= 0` 时直接 return。
  - 不修改 enemy team reward。
- 验证：
  - 新增 `hybrid_arena/minimoba/tests/test_objective_reward.py`
  - 测试：`test_add_team_reward_splits_across_team`、`test_add_team_reward_ignores_non_positive_amount`
  - 命令：`pytest hybrid_arena/minimoba/tests/test_objective_reward.py -v`

## Step F13.3：在结构物伤害路径加入 objective progress reward

- **scope: review**
- 操作：
  - 修改 `hybrid_arena/minimoba/game_engine.py::_damage_structure()`。
  - 保留现有 attacker damage reward：`step_rewards[attacker_id] += reward_config.damage * actual`。
  - 新增 team-shared objective reward：
    - tower：`actual * reward_config.objective_tower_damage_team`
    - base：`actual * reward_config.objective_base_damage_team`
  - 对 team reward 应用 `min(team_amount, objective_step_cap_team)`。
  - 调用 `_add_team_reward(step_rewards, attacker.team, team_amount)`。
  - `objective_enabled=False` 时跳过新增 objective reward。
- 验证：
  - `test_tower_damage_grants_team_objective_reward`
  - `test_base_damage_grants_larger_team_objective_reward_than_tower_damage`
  - `test_objective_step_cap_limits_shared_reward`
  - 命令：`pytest hybrid_arena/minimoba/tests/test_objective_reward.py -v`

## Step F13.4：实现 base exposed 一次性 team reward

- **scope: review**
- 操作：
  - 修改 `hybrid_arena/minimoba/game_engine.py`。
  - 在 `GameState.__init__()` 增加 `self.base_exposed_rewarded: dict[str, bool] = {"red": False, "blue": False}`。
  - 在 `reset()` 重置。
  - 在 `_handle_structure_destroy()` 中，当 tower 被摧毁后调用 `_sync_structure_counts()`。
  - 如果被摧毁 tower 所属队伍的 alive tower count 变为 0：
    - enemy base 暴露。
    - 攻击方团队获得 `reward_config.objective_base_exposed_team`。
    - 对同一 enemy team 只发一次。
- 验证：
  - `test_destroy_last_tower_grants_base_exposed_reward_once`
  - `test_destroy_non_last_tower_does_not_grant_base_exposed_reward`
  - 命令：`pytest hybrid_arena/minimoba/tests/test_objective_reward.py -v`


## Step F13.5：新增 objective 诊断字段到 `GameState`

- **scope: auto**
- 操作：
  - 修改 `hybrid_arena/minimoba/game_engine.py`。
  - 增加 runtime counters：
    - `self.red_tower_damage = 0.0`
    - `self.blue_tower_damage = 0.0`
    - `self.red_base_damage = 0.0`
    - `self.blue_base_damage = 0.0`
  - `reset()` 清零。
  - `_damage_structure()` 中按 attacker team 与 structure type 累加。
- 验证：
  - `test_structure_damage_counters_update`
  - 命令：`pytest hybrid_arena/minimoba/tests/test_objective_reward.py -v`

## Step F13.6：扩展 evaluator objective 指标

- **scope: auto**
- 操作：
  - 修改 `hybrid_arena/training/evaluator.py`。
  - 在每局结束时收集并返回：
    - `avg_tower_damage`
    - `avg_base_damage`
    - `avg_enemy_base_hp_remaining`
    - `base_exposed_rate`
    - `hard_win_rate`
    - `timeout_win_rate`
    - `timeout_draw_rate`
  - 语义：
    - `avg_tower_damage`：red team 对 enemy tower 的平均伤害。
    - `avg_base_damage`：red team 对 enemy base 的平均伤害。
    - `avg_enemy_base_hp_remaining`：blue base 剩余 hp 平均值。
    - `base_exposed_rate`：blue tower 全灭 episode 比例。
  - 保留已有字段，不破坏 CSV 兼容。
- 验证：
  - 更新 `hybrid_arena/training/tests/test_evaluator_metrics.py`
  - 测试：`test_evaluator_contains_objective_shaping_metrics`、`test_base_exposed_rate_in_range`
  - 命令：`pytest hybrid_arena/training/tests/test_evaluator_metrics.py -v`

## Step F13.7：让实验配置能传入 reward shaping 参数

- **scope: review**
- 操作：
  - 修改 `hybrid_arena/scripts/run_ablation.py`。
  - 从 YAML 读取可选字段：
    - `reward.objective_enabled`
    - `reward.objective_tower_damage_team`
    - `reward.objective_base_damage_team`
    - `reward.objective_base_exposed_team`
    - `reward.objective_step_cap_team`
  - 构造 `RewardConfig(**reward_cfg)`。
  - 在训练环境和评估环境中传入相同 `reward_config`。
  - 如果当前 `PPOConfig` 不支持 `reward_config`，修改 `hybrid_arena/algorithms/ppo/config.py` 增加 `reward_config: RewardConfig | None = None`，并在 `Trainer` 创建 env 时透传。
- 验证：
  - 新增 `hybrid_arena/training/tests/test_run_ablation_config.py`
  - 测试：`test_run_ablation_loads_reward_config`、`test_trainer_env_receives_reward_config`
  - 命令：`pytest hybrid_arena/training/tests/test_run_ablation_config.py -v`

## Step F13.8：新增 shaping 专用 sanity 配置

- **scope: auto**
- 操作：
  - 创建 `configs/experiments/sanity_2v2_objective_shaping.yaml`。
  - 内容必须包含：

```yaml
experiment:
  name: sanity_2v2_objective_shaping
  seeds: [42]
  algorithms: [ppo]
  opponents: [random]
  episodes: 30
  max_steps: 200
  map_size: 16
  team_size: 2

training:
  total_timesteps: 100000
  num_steps: 128
  num_envs: 4
  device: cpu

reward:
  objective_enabled: true
  objective_tower_damage_team: 0.001
  objective_base_damage_team: 0.003
  objective_base_exposed_team: 1.0
  objective_step_cap_team: 0.25

outputs:
  checkpoint_dir: checkpoints/sanity_2v2_objective_shaping
  result_dir: results/sanity_2v2_objective_shaping
```

- 验证：
  - `python -m hybrid_arena.scripts.run_ablation --config configs/experiments/sanity_2v2_objective_shaping.yaml --dry-run`
  - dry-run 输出应显示 mode=train_eval、algo=ppo、seed=42、opponent=random、training timesteps=100000。

## Step F13.9：执行 100k 重训并更新文档

- **scope: review**
- 操作：
  - 执行：
    - `python -m hybrid_arena.scripts.run_ablation --config configs/experiments/sanity_2v2_objective_shaping.yaml --mode train_eval`
  - 跑完后执行：
    - `pytest hybrid_arena/minimoba/tests/test_objective_reward.py hybrid_arena/training/tests/test_evaluator_metrics.py hybrid_arena/training/tests/test_run_ablation_config.py -v`
    - `pytest -q`
    - `ruff check hybrid_arena`
  - 更新：
    - `docs/report.md`
    - `docs/issues.md`
    - `docs/progress.md`
    - `docs/experiment-report-v0.md`（如文件存在且当前实验表仍在维护）
- 必须记录：
  - `win_rate`
  - `hard_win_rate`
  - `timeout_win_rate`
  - `timeout_draw_rate`
  - `avg_len`
  - `avg_reward`
  - `avg_tower_damage`
  - `avg_base_damage`
  - `avg_enemy_base_hp_remaining`
  - `base_exposed_rate`
  - `avg_towers_destroyed`
  - `avg_tower_hp_advantage`
- 判定：
  - 成功信号：`hard_win_rate > 0.0`，或 `base_exposed_rate > 0.0` 且 `avg_base_damage > 0.0`。
  - 仍阻塞：`hard_win_rate == 0.0` 且 `base_exposed_rate == 0.0` 且 `avg_base_damage == 0.0`。
- 验收：
  - 不要求本轮 `win_rate >= 0.5`。
  - 本轮核心验收是策略是否开始触达 base objective。
  - 如果 `hard_win_rate > 0`，下一轮才考虑多 seed 验证。
  - 如果 `hard_win_rate == 0` 但 `base_exposed_rate/base_damage` 有明显改善，进入小幅 shaping 参数调整。
  - 如果三者均为 0，返回 Iter，考虑 macro-action curriculum / path-to-objective shaping。


## 新增验收标准

- [ ] `RewardConfig` 支持 objective shaping 参数并可关闭。
- [ ] tower/base damage 产生 capped team-shared objective reward。
- [ ] base exposed 只发一次 team reward。
- [ ] evaluator 输出 objective progress 诊断指标。
- [ ] `run_ablation.py` 能从 YAML 透传 reward config。
- [ ] `sanity_2v2_objective_shaping.yaml` dry-run 正常。
- [ ] 100k 重训完成并记录 hard/timeout/objective 指标。
- [ ] 仍不得执行 300k-500k 长训。

## 风险与控制

| 风险 | 控制 |
|---|---|
| reward hacking：只打结构物不终结 | `hard_win_rate`、`avg_base_damage`、`enemy_base_hp_remaining` 同时监控 |
| shaping reward 压过 combat reward | `objective_step_cap_team=0.25`，且不放大 `win/lose` |
| team-shared reward credit 混乱 | 先只在 2v2 sanity 验证，不扩到 4v4 |
| base 仍不可达 | `base_exposed_rate` 单独作为中间指标 |
| 参数一次调不准 | 本轮只跑一组默认参数，失败后基于指标再 patch |

## 给执行端的合并说明

- 将本文件作为 `docs/inbox/plan.md`。
- merge-back 时保留当前 `docs/plan.md` 原主体，只追加本 Phase F13 和 v2 变更记录。
- 不要覆盖旧模块的历史内容。
- 不要把本轮任务拆成新长期模块；这是 Phase F 训练信号纠偏任务。
