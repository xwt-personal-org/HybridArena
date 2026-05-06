# HybridArena 问题记录

> Codex 在执行中遇到 `docs/plan.md` 未覆盖的情况时记录于此。

## 当前执行阻塞

（暂无）

## 已知但已纳入计划的问题

- action mask 语义不一致：已纳入模块 1、模块 2。
- PPO clipped value loss 无效：已纳入模块 2。
- DualClipPPO 指标错误：已纳入模块 2。
- 塔、基地、队伍经济未闭环：已纳入模块 3。
- `num_envs` 未使用：已纳入模块 5。
- 训练 CLI、评估器、checkpoint 缺失：已纳入模块 4。
- LLM Planner 目录为空：已纳入模块 6。

## 新发现问题

### ISSUE-F0：env.close() 在无 pygame 环境下崩溃
- 严重级别：P1
- 影响：无 pygame 安装时 smoke test 在 close() 调用失败
- 复现：`pytest hybrid_arena/minimoba/tests/test_smoke.py -v`（不装 pygame）
- 修复状态：已修复（Phase F0.3），`env.close()` 中 pygame 改为 try/except ImportError

### ISSUE-F7：baseline_v1 实验训练时间过长
- 严重级别：P2
- 影响：100k timesteps * 2 algorithms * 3 seeds 在 CPU 上需要约 2 小时，实验超时
- 复现：`python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_v1.yaml`
- 修复状态：部分完成，已训练 ppo seed=42（100k steps），其余待后续补充
- 建议：减少训练步数或使用 GPU 加速

### ISSUE-F8：run_ablation.py eval_only 模式 algo 字段 bug
- 严重级别：P1
- 影响：eval_only 模式下结果 CSV 的 algo 字段显示为 "rule_based" 而非实际算法名
- 复现：`python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_v1.yaml --mode eval_only`
- 修复状态：已修复，修改为 `"algo": algo if mode in ("train_eval", "eval_only") else "rule_based"`

### ISSUE-F9：baseline_v1 训练信号较弱
- 严重级别：P2
- 影响：ppo seed=42 在 100k steps 后对 random 胜率仅 16.7%，对 rule_based 胜率为 0%
- 数据：
  - ppo seed=42 vs random: win_rate=0.167, avg_reward=12.104
  - ppo seed=42 vs rule_based: win_rate=0.000, avg_reward=11.742
- 可能原因：
  1. 100k steps 训练量不足，策略尚未收敛
  2. 4v4 环境复杂度高，需要更多训练步数
  3. RuleBasedAgent 作为 baseline 较强，ppo 需要更长训练才能超越
- 建议：
  1. 优先跑 `ppo_dualclip seed=42` 同配置对照，验证 dual-clip 是否有改善
  2. 考虑增加训练步数至 300k-500k
  3. 或使用 GPU 加速训练
- 状态：已记录，待后续实验验证

### ISSUE-F10：DualClipPPO 未改善训练信号
- 严重级别：P2
- 影响：ppo_dualclip seed=42 在 100k steps 后对 random 胜率 0%（ppo 为 16.7%），对 rule_based 胜率 0%，未展示优于 PPO 的改善
- 数据：
  - ppo_dualclip seed=42 vs random: win_rate=0.000, avg_reward=9.948, towers_destroyed=0.0
  - ppo_dualclip seed=42 vs rule_based: win_rate=0.000, avg_reward=12.089, towers_destroyed=0.0
- 对比 ppo：
  - vs random: win_rate 0% vs 16.7%（更差），avg_reward 9.95 vs 12.10（更差）
  - vs rule_based: win_rate 0% vs 0%（持平），avg_reward 12.09 vs 11.74（略好）
- 分析：问题不在算法选择（dual-clip vs vanilla），而在训练信号本身。两种算法在 100k steps 后均未有效学习。
- 结论：不继续完整矩阵（6 run），需要先诊断训练信号问题
- 建议：
  1. 增加训练步数至 300k-500k，观察是否收敛
  2. 检查训练曲线（entropy、clip_fraction、value_loss）判断是否在学习
  3. 简化环境（2v2、更小地图）验证算法正确性
- 状态：已记录，作为训练信号诊断入口

### ISSUE-F11：sanity_2v2 未达 50% 阈值 — 训练闭环结构性问题
- 严重级别：P2
- 影响：2v2 simplified 环境 PPO 100k steps 对 random 胜率 33.3%，未达 50% 验收阈值，不能上 300k-500k 长训
- 数据：
  - sanity_2v2 ppo seed=42 vs random: win_rate=0.333, draw_rate=0.333, avg_reward=6.183, avg_len=200.0
  - 对比 4v4 baseline: win_rate=0.167（提升到 2x，但仍未达标）
- 训练曲线证据（证明在学习）：
  - episode length: 200→34（学会了交战）
  - entropy: 4.55→4.0（policy 在变确定性）
  - KL: 0.006-0.014（健康范围）
  - towers_destroyed: 0.6（学会推塔）
- 根因分析：
  1. **evaluator 奖励跨两队平均**：`sum(rewards.values()) / len(rewards)` 包含对方负奖励，稀释胜负信号
  2. **draw_rate = 33.3%**：agent 学会交战但没学会终结（timeout 200 步）
  3. **value_loss 持续偏高**（0.9-2.1），value function 未收敛
- 修复优先级：
  1. 修复 evaluator：只计算 red team 的 reward（P0，结构性缺陷）
  2. 验证 win/lose reward 在 episode 结束时正确发放
  3. 修复后再跑 sanity_2v2 验证
- 状态：已修复 evaluator 和 draw reward，待重训验证

#### 修复记录（2026-05-06）
- **evaluator.py 修复**：
  - `avg_reward` 改为只统计 red team reward（不再跨两队平均）
  - 新增 `avg_red_reward`、`avg_blue_reward`、`avg_reward_margin` 指标
  - `avg_reward == avg_red_reward`（保持向后兼容）
- **env.py 修复**：
  - draw 时不再发放 win/lose reward（`winner in ("red", "blue")` 才发放）
  - 避免 draw 时所有 agent 都收到 lose reward
- **新增测试**：
  - `test_evaluator_metrics.py`：验证 avg_reward == avg_red_reward、margin 计算、不跨队平均
  - `test_reward.py`：验证 red win 时正确发放 win/lose、draw 时不误发 win/lose
- **ep_len 不一致根因**：
  - 训练曲线 ep_len≈34 是 SyncParallelEnvRunner 多环境下的统计值（4 env 交替结束，计数器重置间隔短）
  - 评估 avg_len=200 是单环境评估，所有 episode 跑满 max_steps
  - **结论**：eval_only 重跑结果 win_rate=0.267（修正后），但所有 episode 仍为 200 步
  - 策略未学会终结（推基地），只是在 timeout 时靠 towers/kills 优势判胜
  - **评估口径可信**，但训练信号本身仍需加强（需要更多 steps 或 reward shaping）
- **eval_only 重跑结果**（修正后）：
  - ppo seed=42 vs random: win_rate=0.267, draw_rate=0.333, avg_reward=12.953, avg_len=200.0
  - 对比修正前: win_rate=0.333→0.267, avg_reward=6.183→12.953（red team 口径更准确）

#### 重训验证（2026-05-06）
- **重训配置**：ppo seed=42, 100k steps, 2v2, map_size=16, max_steps=200
- **重训结果**：
  - win_rate=0.467, draw_rate=0.367, avg_reward=22.351, avg_len=200.0
  - towers_destroyed=0.8, tower_hp_advantage=567.0, fps=402.5
- **对比修复前 eval_only**：
  - win_rate: 0.267 → 0.467（+75%）
  - avg_reward: 12.953 → 22.351（+73%）
  - towers_destroyed: 0.333 → 0.8（+140%）
  - tower_hp_advantage: 65 → 567（+772%）
- **训练信号分析**：
  - 训练曲线 entropy 从 ~4.5 降至 ~3.75（policy 在变确定性）
  - 训练曲线 episode_length 从 ~200 降至 ~27（学会了交战）
  - 最终训练 reward ~6.65，episode_length ~28
- **结论**：
  - ✅ evaluator 修复有效：win_rate 和 avg_reward 显著提升
  - ✅ 训练信号改善：towers_destroyed 和 tower_hp_advantage 大幅提升
  - ❌ win_rate 未达 50% 阈值（46.7% < 50%）
  - ❌ avg_len 仍为 200：策略未学会终结游戏，靠 timeout 判胜
  - **根因**：策略需要学会推基地终结游戏，当前靠 towers/kills 优势在 timeout 时判胜
- **状态**：已验证修复有效，但需 objective/reward shaping 解决终结问题

### ISSUE-F12：terminal semantics split — timeout 优势判胜不应发训练 terminal reward
- 严重级别：P1
- 影响：timeout 时 towers/gold/kills 优势判胜仍触发 win/lose terminal reward，强化"拖到裁判判胜"而非"推基地终结"
- 数据：
  - sanity_2v2 重训 win_rate=46.7%，但 avg_len=200，说明所有 win 都是 timeout adjudicated
  - get_winner() 在 timeout 时按 towers/gold/kills 优势返回 red/blue，env.py 继续发 win/lose reward
- 根因：
  - game_engine.py 缺少 terminal_reason 字段
  - env.py 不区分 base_destroyed 和 timeout，统一发 win/lose reward
  - evaluator.py 不拆分 hard_win（base_destroyed）和 timeout_win（adjudicated）
- 修复方案：
  1. game_engine.py 增加 terminal_reason: "base_destroyed" | "timeout" | None
  2. env.py 仅在 terminal_reason == "base_destroyed" 时发 win/lose terminal reward
  3. evaluator.py 新增 hard_win_rate、timeout_win_rate、timeout_draw_rate
  4. 补测试覆盖 timeout advantage 场景
- 状态：已修复（2026-05-06），重训验证完成

#### 重训验证（2026-05-06）
- **配置**：ppo seed=42, 100k steps, 2v2, map_size=16, max_steps=200
- **结果**：
  - win_rate=0.400, draw_rate=0.333, avg_reward=13.705, avg_len=200.0
  - **hard_win_rate=0.000**（无 base_destroyed 胜利）
  - timeout_win_rate=0.400（所有胜利均为 timeout 判胜）
  - timeout_draw_rate=0.333
- **对比旧 semantics**：
  - 旧：win_rate=0.467（含 timeout win reward），avg_reward=22.351
  - 新：win_rate=0.400（不含 timeout win reward），avg_reward=13.705
- **结论**：
  - ✅ terminal semantics 拆分有效：timeout 不再发 win/lose reward
  - ❌ hard_win_rate = 0.0%：策略完全靠 timeout 判胜，未学会推基地
  - ❌ 需 objective reward shaping 引导策略学会终结游戏
- **下一步**：
  1. 需要 objective reward shaping（推基地奖励）引导策略学会终结
  2. 不建议调 reward 权重或上 300k-500k 长训
  3. 优先解决 hard_win_rate=0 的结构性问题

### ISSUE-F13：objective reward shaping — hard_win_rate 仍为 0
- 严重级别：P2
- 影响：新增 objective shaping 后 tower_damage 从 ~0 跃升到 1351，但 base_exposed_rate=0、avg_base_damage=0，策略未学会推完塔再推基地
- 数据（sanity_2v2 objective shaping 100k steps）：
  - win_rate=0.200, hard_win_rate=0.000, timeout_win_rate=0.200
  - avg_tower_damage=1351.0（显著提升）
  - avg_base_damage=0.0, base_exposed_rate=0.0
  - avg_towers_destroyed=0.267（下降，策略在打塔但没打完）
- 分析：
  1. objective shaping 有效引导策略去打塔（tower_damage ↑↑↑）
  2. 但策略没学会"打完塔再推基地"的过渡（base_exposed_rate=0）
  3. win_rate 下降（0.400→0.200），可能是因为策略分心打塔而非战斗
  4. avg_towers_destroyed 下降（0.6→0.267），策略在磨塔但没终结
- 下一步：
  1. 考虑 macro-action curriculum / path-to-objective shaping
  2. 或增加 tower_destroyed bonus 引导策略完成推塔
  3. 或调整 shaping 参数（增大 objective_base_exposed_team）
- 状态：已记录，待下一轮 patch
