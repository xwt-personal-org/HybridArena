# Execution Report

## STATUS: NEEDS_ESCALATION

> 上次更新：2026-05-07 | plan.md 版本：v2

## Last Execution
- 来源：dispatch:fix（Phase F13 review report 修复）
- 摘要：按 review report 修复 H-1/H-2/H-3/M-1/M-4。修正 report 状态、Trainer eval 口径、默认 shaping 开关、YAML key 校验、补 objective closure 端到端测试。

## Completed
- [x] F13.1: `RewardConfig` 新增 objective shaping 字段（objective_enabled/tower_damage_team/base_damage_team/base_exposed_team/step_cap_team）
- [x] F13.2: `_add_team_reward` helper（game_engine.py）
- [x] F13.3: 结构物伤害路径加入 objective progress reward
- [x] F13.4: base exposed 一次性 team reward
- [x] F13.5: objective 诊断字段（red/blue_tower/base_damage, base_exposed_rewarded）
- [x] F13.6: evaluator 新增 avg_tower_damage/avg_base_damage/avg_enemy_base_hp_remaining/base_exposed_rate
- [x] F13.7: run_ablation 透传 reward config（PPOConfig + Trainer + evaluate_policy）
- [x] F13.8: `sanity_2v2_objective_shaping.yaml`
- [x] F13.9: 100k 重训（2039s）+ 138 tests passed + ruff clean

## In Review
（无）

## Blocked
- [ ] hard_win_rate=0.0% — objective shaping 引导了 tower_damage 但未引导 base destruction
- [ ] base_exposed_rate=0.0% — 策略没学会推完塔
- [ ] avg_base_damage=0.0% — 基地从未被攻击
- [ ] sanity_2v2 未达 50% 阈值
- [ ] 300k-500k 长训 — 需 scripted objective policy 能稳定打出 base_exposed_rate>0 后再上

## Discovered Issues
- ISSUE-F13: objective reward shaping — tower_damage 提升但 base_exposed_rate=0
- tower_damage=1351（显著提升），但 avg_towers_destroyed=0.267（下降）
- 策略在磨塔但没学会终结
- 需架构级调整 reward/action/curriculum，不得进入长训

## Recommendations
- 不得进入 300k-500k 长训，需架构级调整
- 考虑 macro-action curriculum / tower_destroyed team bonus / path-to-objective shaping
- 先让 scripted objective policy 能稳定打出 base_exposed_rate>0 和 avg_base_damage>0

## 实验结果

### sanity_2v2 objective shaping（100k steps）
| algo | seed | opponent | win_rate | draw_rate | hard_win | timeout_win | timeout_draw | avg_reward | avg_len | towers | tower_hp | tower_dmg | base_dmg | enemy_base_hp | base_exp | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.200 | 0.333 | 0.000 | 0.200 | 0.333 | 14.562 | 200.0 | 0.3 | -21.0 | 1351.0 | 0.0 | 2000.0 | 0.000 | 403.7 |

### sanity_2v2 无 shaping（100k steps，新 terminal semantics）
| algo | seed | opponent | win_rate | draw_rate | hard_win | timeout_win | timeout_draw | avg_reward | avg_len | towers | tower_hp | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.400 | 0.333 | 0.000 | 0.400 | 0.333 | 13.705 | 200.0 | 0.6 | -185.0 | 412.2 |

### sanity_2v2 旧 terminal semantics（100k steps）
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
