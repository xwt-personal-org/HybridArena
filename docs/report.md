# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新：2026-05-06 | plan.md 版本：v1

## Last Execution
- 来源：dispatch:patch（sanity_2v2 重训验证）
- 摘要：使用修复后的 evaluator/env 重训 sanity_2v2 ppo seed=42 100k steps。win_rate=46.7%（修复前 eval_only 26.7%），avg_reward=22.35（修复前 12.95），avg_len 仍为 200。未达 50% 阈值，记录为 objective/reward shaping 问题。

## Completed
- [x] 修复 `evaluator.py`：`avg_reward` 只统计 red team
- [x] 新增指标：`avg_red_reward`、`avg_blue_reward`、`avg_reward_margin`
- [x] 修复 `env.py`：draw 时不发放 win/lose reward
- [x] 新增 evaluator 单测 + env 终局 reward 单测
- [x] 排查 ep_len 不一致根因（多环境统计 vs 单环境评估）
- [x] 重跑 sanity_2v2 eval_only 验证：win_rate=0.267（修正后口径）
- [x] 重跑 sanity_2v2 train+eval 验证：win_rate=0.467（修复后重训）

## In Review
- [ ] sanity_2v2 重训结果 — win_rate=46.7% 接近但未达 50% 阈值

## Blocked
- [ ] sanity_2v2 未达 50% 阈值 — avg_len=200 说明策略未学会终结，需 objective/reward shaping
- [ ] 300k-500k 长训 — 需 sanity_2v2 达到 50% 后再上

## Discovered Issues
- 修复后重训 win_rate 提升：26.7% → 46.7%（+75%），avg_reward 提升：12.95 → 22.35（+73%）
- avg_len 仍为 200：策略靠 timeout 时 towers/kills 优势判胜，未学会推基地
- towers_destroyed 从 0.33 → 0.8（+140%），tower_hp_advantage 从 65 → 567（+772%）
- **结论**：evaluator 修复有效，训练信号改善，但策略未学会终结游戏

## Recommendations
- 不建议直接调 reward 权重，本轮只验证修复后训练信号
- 下一步：增加 objective reward shaping（如推塔进度奖励、基地接近奖励）或增加训练步数
- avg_len=200 是核心问题：策略需要学会终结游戏而非靠 timeout 判胜

## 实验结果

### sanity_2v2（修复后重训）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.467 | 0.367 | 22.351 | 200.0 | 0.8 | 567.0 | 402.5 |

### sanity_2v2（修复前 eval_only，旧 checkpoint）
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
