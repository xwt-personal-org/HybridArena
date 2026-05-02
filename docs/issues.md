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
