# Execution Report

## STATUS: ALL_CLEAR

> 上次更新：2026-05-03 | plan.md 版本：v1

## Last Execution
- 来源：dispatch:patch
- 摘要：运行 ppo_dualclip seed=42 对照实验（100k steps），评估 vs random（0%）和 vs rule_based（0%），未改善训练信号，记录为诊断入口

## Completed
- [x] 训练 ppo_dualclip seed=42（100k steps, CPU ~72 分钟）
- [x] 评估 ppo_dualclip seed=42 vs random → win_rate=0.0%, avg_reward=9.95
- [x] 评估 ppo_dualclip seed=42 vs rule_based → win_rate=0.0%, avg_reward=12.09
- [x] 更新 baseline_v1 partial 结果表（新增 2 行 ppo_dualclip 数据）
- [x] 更新 `docs/experiment-report-v0.md`
- [x] 更新 README 实验结果表
- [x] 更新 `docs/progress.md`（新增模块 F9）
- [x] 更新 `docs/issues.md`（新增 ISSUE-F10）

## In Review
- [ ] baseline_v1 完整矩阵 — DualClipPPO 未改善，暂停扩展

## Blocked
- [ ] baseline_v1 完整实验 — 训练信号问题待诊断后再决定是否继续

## Discovered Issues
- ISSUE-F10：DualClipPPO 未改善训练信号（P2）— 对 random 0% vs ppo 16.7%，对 rule_based 0% vs 0%

## Recommendations
- 诊断训练信号问题：增加训练步数至 300k-500k / 简化环境验证 / 检查训练曲线
- DualClipPPO 在 100k steps 下不优于 PPO，问题可能在训练量或环境复杂度

## 实验结果

### baseline_v1（部分完成）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.167 | 0.033 | 12.104 | 493.0 | 0.4 | -1187.0 | 375.9 |
| ppo | 42 | rule_based | 0.000 | 0.067 | 11.742 | 500.0 | 0.4 | -663.0 | 372.7 |
| ppo_dualclip | 42 | random | 0.000 | 0.033 | 9.948 | 498.6 | 0.0 | -2277.0 | 387.8 |
| ppo_dualclip | 42 | rule_based | 0.000 | 0.000 | 12.089 | 500.0 | 0.0 | -1985.0 | 387.9 |
