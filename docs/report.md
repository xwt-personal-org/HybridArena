# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新：2026-05-06 | plan.md 版本：v1

## Last Execution
- 来源：dispatch:fix（ISSUE-F12: terminal semantics split）
- 摘要：按 review report 修复 H-1/M-1/M-3。拆分终局原因（base_destroyed vs timeout），timeout 不发 win/lose terminal reward，evaluator 新增 hard_win_rate/timeout_win_rate/timeout_draw_rate 指标拆分，补全测试覆盖。

## Completed
- [x] H-1: `game_engine.py` 增加 `terminal_reason` 字段（"base_destroyed"/"timeout"/None）
- [x] H-1: `env.py` 仅在 `terminal_reason == "base_destroyed"` 时发 win/lose terminal reward
- [x] M-1: `evaluator.py` 新增 `hard_win_rate`、`timeout_win_rate`、`timeout_draw_rate`、`hard_red/blue_wins`、`timeout_red/blue_wins`、`timeout_draws`
- [x] M-3: 补 `test_timeout_advantage_does_not_grant_win_lose_reward`、`test_base_destroyed_grants_win_lose_reward`
- [x] M-3: 补 evaluator 指标拆分测试（`test_evaluator_splits_hard_win_and_timeout_win`、`test_hard_win_rate_is_base_destroyed_only`）
- [x] 18 tests passed, ruff clean

## In Review
- [ ] terminal semantics split — 需重训验证 hard_win_rate vs timeout_win_rate 分布

## Blocked
- [ ] sanity_2v2 未达 50% 阈值 — terminal semantics 拆分后需重训验证
- [ ] 300k-500k 长训 — 需 sanity_2v2 达到 50% 后再上

## Discovered Issues
- ISSUE-F12: terminal semantics split — timeout 优势判胜不应发训练 terminal reward
- 修复前：timeout adjudicated win 也会触发 win/lose reward，强化"拖到裁判判胜"
- 修复后：只有 base_destroyed 才发 win/lose reward，timeout 胜负仅用于评估 adjudication

## Recommendations
- 重跑 sanity_2v2 100k 验证 hard_win_rate vs timeout_win_rate 分布
- 若 hard_win_rate 接近 0%，说明策略完全靠 timeout 判胜，需 objective reward shaping
- 不建议调 reward 权重，不建议上 300k-500k 长训

## 实验结果

### sanity_2v2（修复后重训，旧 terminal semantics）
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
