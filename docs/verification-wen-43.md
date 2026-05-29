# WEN-43 验证报告：MiniMOBA 环境与 RL 算法主线稳定化

## 验证目的

确认 MiniMOBA 环境、动作空间、action mask、训练/评估 CLI 与主线算法测试在当前主分支基线上保持可用。

## 验收对照

| 验收项 | 状态 | 证据 |
|--------|------|------|
| `pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v` 通过 | [x] PASS | 105 passed, 1 skipped, 130.06s |
| train/evaluate smoke 命令可执行或失败原因明确 | [x] PASS | train/evaluate 均执行成功 |
| 训练指标与 checkpoint/report 输出路径记录清楚 | [x] PASS | 见下方输出路径 |

## 验证命令

```bash
pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v
```

结果：`105 passed, 1 skipped in 130.06s`。

```bash
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --total-timesteps 512 --num-steps 32 --device cpu --checkpoint-dir artifacts/wen43_train_smoke/checkpoints --save-interval 1
```

结果：训练成功，`512` steps，耗时约 `71s`。

```bash
python -m hybrid_arena.scripts.evaluate --checkpoint artifacts/wen43_train_smoke/checkpoints/ppo_dualclip_seed42_step512.pt --algorithm ppo_dualclip --opponent rule_based --episodes 3 --seed 42 --device cpu --output artifacts/wen43_eval_smoke/eval_report.json
```

结果：评估成功，`red_wins=0`、`blue_wins=3`、`draws=0`、`avg_reward=+62.538`、`avg_length=899.3`。

```bash
python -m hybrid_arena.scripts.run_ablation --dry-run --episodes 1 --max-steps 32 --output-dir artifacts/wen43_ablation_smoke
```

结果：dry-run 成功，计划 `12` 个 baseline runs。

## 输出路径

| 类型 | 路径 |
|------|------|
| checkpoint | `artifacts/wen43_train_smoke/checkpoints/ppo_dualclip_seed42_step512.pt` |
| 训练结果 | `artifacts/wen43_train_smoke/checkpoints/ppo_dualclip_seed42_results.pkl` |
| 评估报告 | `artifacts/wen43_eval_smoke/eval_report.json` |
| ablation dry-run 输出目录 | `artifacts/wen43_ablation_smoke` |

## 依赖与边界

- 远端默认主分支为 `master`，不存在 `origin/main`。
- WEN-42 远端分支尚未合入 `origin/master`；本轮按当前主分支基线验证 WEN-43。
- ISSUE-F13 仍是训练有效性阻塞：当前验证确认 scripted objective/base 可达测试通过，但不声明 RL 已学会 hard win。
