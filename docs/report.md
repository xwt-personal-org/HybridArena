# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新：2026-05-06 | plan.md 版本：v1

## Last Execution
- 来源：dispatch:fix (ISSUE-F11)
- 摘要：修复 evaluator 跨两队平均奖励、draw 时误发 win/lose reward，新增 avg_red/blue/margin 指标，加单测，排查 ep_len 不一致根因，重跑 eval_only 验证。

## Completed
- [x] 修复 `evaluator.py`：`avg_reward` 只统计 red team（`evaluator.py:81` 跨两队平均 → 按 team 分别累加）
- [x] 新增指标：`avg_red_reward`、`avg_blue_reward`、`avg_reward_margin`
- [x] 修复 `env.py`：draw 时不发放 win/lose reward（`winner in ("red", "blue")` 才发放）
- [x] 新增 evaluator 单测（`test_evaluator_metrics.py`）：验证 avg_reward==avg_red_reward、margin 计算、不跨队平均
- [x] 新增 env 终局 reward 单测（`test_reward.py`）：red win 发放 win/lose、draw 不误发
- [x] 排查 ep_len≈34 vs avg_len=200 根因：训练用 SyncParallelEnvRunner（4 env 交替结束，计数器间隔短），评估用单 env（跑满 200 步）
- [x] 重跑 sanity_2v2 eval_only：win_rate=0.267, avg_reward=12.953（修正后口径）

## In Review
- [ ] evaluator 修复验证 — 需确认修正后指标可信
- [ ] ep_len 不一致是否影响训练诊断 — 已记录根因，不影响评估口径

## Blocked
- [ ] sanity_2v2 未达 50% 阈值 — eval 口径可信后仍需加强训练信号（更多 steps / reward shaping）
- [ ] 300k-500k 长训 — 需 sanity_2v2 达到 50% 后再上

## Discovered Issues
- ISSUE-F11 修复：evaluator 跨两队平均已修正，draw reward 已修正
- 评估口径可信，但策略未学会终结（推基地），靠 timeout 时 towers/kills 优势判胜
- 训练曲线 ep_len≈34 是多环境统计假象，单环境评估均为 200 步

## Recommendations
- eval 口径已可信，可重跑 100k sanity_2v2 验证训练信号
- 若重训后 win_rate 仍 < 50%，考虑增加 reward shaping（如推塔进度奖励）
- 不建议直接上 300k-500k 长训，先在 100k 验证信号

## 实验结果

### sanity_2v2（修正后 eval_only）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | avg_red_reward | avg_blue_reward | avg_reward_margin |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.267 | 0.333 | 12.953 | 200.0 | 12.953 | -6.008 | 18.961 |

### baseline_v1（部分完成，修正前口径）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.167 | 0.033 | 12.104 | 493.0 | 0.4 | -1187.0 | 375.9 |
| ppo | 42 | rule_based | 0.000 | 0.067 | 11.742 | 500.0 | 0.4 | -663.0 | 372.7 |
| ppo_dualclip | 42 | random | 0.000 | 0.033 | 9.948 | 498.6 | 0.0 | -2277.0 | 387.8 |
| ppo_dualclip | 42 | rule_based | 0.000 | 0.000 | 12.089 | 500.0 | 0.0 | -1985.0 | 387.9 |
