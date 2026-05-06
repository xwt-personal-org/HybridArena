# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新：2026-05-03 | plan.md 版本：v1

## Last Execution
- 来源：dispatch:patch
- 摘要：从 ISSUE-F10 开始训练信号诊断，新增 2v2 simplified sanity 配置，训练 PPO 100k steps 并评估。win_rate=33.3%（4v4 baseline 16.7%），证明环境复杂度是因素之一，但未达 50% 阈值。发现 evaluator 奖励跨两队平均的结构性缺陷（ISSUE-F11）。

## Completed
- [x] 创建 `configs/experiments/sanity_2v2.yaml`（2v2, map_size=16, max_steps=200）
- [x] 训练 PPO seed=42, 100k steps（2v2 simplified, 2069s, ~34.5 分钟）
- [x] 评估 vs random, 30 episodes → win_rate=33.3%, draw_rate=33.3%, avg_reward=6.183
- [x] 导出训练曲线 CSV（781 数据点，含 policy_loss/value_loss/entropy/clip_fraction/kl）
- [x] 对比 4v4 baseline：win_rate 16.7%→33.3%，episode_length 493→34
- [x] 诊断训练闭环结构性问题：evaluator 跨两队平均、draw_rate 33% timeout
- [x] 更新 `docs/experiment-report-v0.md`（新增 sanity_2v2 章节）
- [x] 更新 `docs/issues.md`（新增 ISSUE-F11）
- [x] 修改 Trainer 支持 training_curve 导出（`_log_step` 累积 + train() 返回 + CSV 输出）

## In Review
- [ ] sanity_2v2 结果分析 — 33.3% 未达 50% 阈值，建议先修 evaluator 再重跑

## Blocked
- [ ] 300k-500k 长训 — 需 sanity_2v2 达到 50% 后再上
- [ ] baseline_v1 完整矩阵 — 训练信号问题待解决

## Discovered Issues
- ISSUE-F11：sanity_2v2 未达 50% 阈值（P2）— evaluator 奖励跨两队平均 + draw_rate 33% timeout
- 训练曲线正面信号：episode_length 200→34、entropy 下降、KL 健康
- 训练曲线问题信号：value_loss 持续偏高（0.9-2.1）、avg_reward 随 episode 缩短下降

## Recommendations
- **P0**：修复 evaluator，只计算 red team 的 reward（当前跨两队平均稀释胜负信号）
- **P0**：验证 win/lose reward 在 episode 结束时正确发放
- 修复后重跑 sanity_2v2，目标 win_rate ≥ 50%
- 达标后再考虑 4v4 300k-500k 长训

## 实验结果

### baseline_v1（部分完成）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.167 | 0.033 | 12.104 | 493.0 | 0.4 | -1187.0 | 375.9 |
| ppo | 42 | rule_based | 0.000 | 0.067 | 11.742 | 500.0 | 0.4 | -663.0 | 372.7 |
| ppo_dualclip | 42 | random | 0.000 | 0.033 | 9.948 | 498.6 | 0.0 | -2277.0 | 387.8 |
| ppo_dualclip | 42 | rule_based | 0.000 | 0.000 | 12.089 | 500.0 | 0.0 | -1985.0 | 387.9 |

### sanity_2v2（训练信号诊断）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.333 | 0.333 | 6.183 | 200.0 | 0.6 | 120.0 | 369.6 |
