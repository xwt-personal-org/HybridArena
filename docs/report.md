# Execution Report

## STATUS: ALL_CLEAR

> 上次更新：2026-05-06 | plan.md 版本：v1

## Last Execution
- 来源：口头指令（ISSUE-F12 重训验证）
- 摘要：重训 sanity_2v2 验证 terminal semantics 修复效果。hard_win_rate=0.000，所有胜利均为 timeout 判胜，策略未学会推基地终结。

## Completed
- [x] H-1: `game_engine.py` 增加 `terminal_reason` 字段（"base_destroyed"/"timeout"/None）
- [x] H-1: `env.py` 仅在 `terminal_reason == "base_destroyed"` 时发 win/lose terminal reward
- [x] M-1: `evaluator.py` 新增 `hard_win_rate`、`timeout_win_rate`、`timeout_draw_rate` 指标
- [x] M-1: `run_ablation.py` CSV/summary 导出新指标列
- [x] M-3: 补测试（18 tests passed，ruff clean）
- [x] 重训 sanity_2v2（100k steps，1992s）→ hard_win_rate=0.000, timeout_win_rate=0.400

## In Review
（无）

## Blocked
- [ ] hard_win_rate=0.0% — 策略完全靠 timeout 判胜，需 objective reward shaping
- [ ] sanity_2v2 未达 50% 阈值 — 需解决 hard_win_rate=0 后重训
- [ ] 300k-500k 长训 — 需 sanity_2v2 达到 50% 后再上

## Discovered Issues
- ISSUE-F12: terminal semantics split — 已修复并重训验证
- **关键发现**：hard_win_rate=0.000，策略未学会推基地终结，所有胜利来自 timeout adjudication

## Recommendations
- 需 objective reward shaping 引导策略学会推基地终结
- 不建议调 reward 权重或上 300k-500k 长训
- 优先解决 hard_win_rate=0 的结构性问题

## 实验结果

### sanity_2v2（新 terminal semantics，重训 100k steps）
| algo | seed | opponent | win_rate | draw_rate | hard_win | timeout_win | timeout_draw | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.400 | 0.333 | 0.000 | 0.400 | 0.333 | 13.705 | 200.0 | 0.6 | -185.0 | 412.2 |

### sanity_2v2（旧 terminal semantics，重训 100k steps）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.467 | 0.367 | 22.351 | 200.0 | 0.8 | 567.0 | 402.5 |

### baseline_v1（部分完成，修正前口径）
| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | towers_destroyed | tower_hp_adv | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.167 | 0.033 | 12.104 | 493.0 | 0.4 | -1187.0 | 375.9 |
| ppo | 42 | rule_based | 0.000 | 0.067 | 11.742 | 500.0 | 0.4 | -663.0 | 372.7 |
| ppo_dualclip | 42 | random | 0.000 | 0.033 | 9.948 | 498.6 | 0.0 | -2277.0 | 387.8 |
| ppo_dualclip | 42 | rule_based | 0.000 | 0.000 | 12.089 | 500.0 | 0.0 | -1985.0 | 387.9 |
