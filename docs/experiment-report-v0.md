# HybridArena 实验报告 v0

> **警告：当前实验不是正式 benchmark。** 以下所有结果为 smoke 参数下验证流水线所用，不代表正式算法结论。正式实验矩阵见下方。

## 环境设置

- Python: `3.10 <= version < 3.13`
- 安装: `pip install -e ".[dev,rl]"`
- 基础验证: `python -m compileall hybrid_arena`、`pytest hybrid_arena/minimoba/tests -v`

## Smoke 参数

| 参数 | smoke 值 | 正式实验值 |
|------|----------|------------|
| episodes | 1 | 30 |
| max_steps | 20–50 | 500 |
| seeds | 42, 123, 456 | 42, 123, 456 |
| opponents | random, rule_based | random, rule_based |
| total_timesteps | 512 | 100000 |
| num_envs | 1 | 4 |

## PPO vs DualClipPPO Smoke 实验

当前 smoke 命令用于验证训练闭环、checkpoint 和 evaluator 可运行：

```bash
python -m hybrid_arena.scripts.train --algo ppo --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
```

## RuleBased vs Random Baseline

```bash
python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 3 --seed 42 --output results/eval_smoke.json
```

## 已知问题

- smoke 训练只证明端到端可执行，不证明策略收敛。
- 当前 LLM Planner MVP 离线可测，不依赖真实外部 LLM API。
- GRPO/QLoRA 需要 planner trace 数据后再进入下一阶段。

## 下一步正式实验矩阵

### baseline_v1

| 参数 | 值 |
|------|-----|
| algorithms | ppo, ppo_dualclip |
| seeds | 42, 123, 456 |
| opponents | random, rule_based |
| episodes | 30 |
| max_steps | 500 |
| total_timesteps | 100000 |
| num_envs | 4 |
| device | cpu |

命令：

```bash
python -m hybrid_arena.scripts.run_ablation --config configs/experiments/baseline_v1.yaml
```

### 计划对比维度

| algo | seeds | opponents | metrics |
|---|---|---|---|
| ppo | 42, 123, 456 | random, rule_based | win_rate, draw_rate, avg_reward, avg_len, avg_towers_destroyed, avg_tower_hp_advantage, fps |
| ppo_dualclip | 42, 123, 456 | random, rule_based | win_rate, draw_rate, avg_reward, avg_len, avg_towers_destroyed, avg_tower_hp_advantage, fps |

---

## baseline_v1 Partial 结果

> **状态：partial** — 仅完成 ppo seed=42 和 ppo_dualclip seed=42（各 100k steps），其余 seed 待后续补充。
>
> **训练环境：** CPU (约 23-24 FPS)，单 run 训练约 40-72 分钟。

### 实验配置

- 配置文件：`configs/experiments/baseline_v1.yaml`
- 训练步数：100,000
- 评估 episodes：30
- max_steps：500
- 训练环境：map_size=32, team_size=4, num_envs=4

### 结果

| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | avg_towers_destroyed | avg_tower_hp_advantage | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.167 | 0.033 | 12.104 | 493.0 | 0.4 | -1187.0 | 375.9 |
| ppo | 42 | rule_based | 0.000 | 0.067 | 11.742 | 500.0 | 0.4 | -663.0 | 372.7 |
| ppo_dualclip | 42 | random | 0.000 | 0.033 | 9.948 | 498.6 | 0.0 | -2277.0 | 387.8 |
| ppo_dualclip | 42 | rule_based | 0.000 | 0.000 | 12.089 | 500.0 | 0.0 | -1985.0 | 387.9 |

### 分析

- **DualClipPPO 未改善训练信号**：ppo_dualclip seed=42 在 100k steps 后对 random 胜率 0%（ppo 为 16.7%），对 rule_based 胜率仍为 0%。
- **DualClipPPO 表现更差**：对 random 的 avg_reward（9.95）低于 ppo（12.10），avg_towers_destroyed 为 0（ppo 为 0.4）。
- **训练信号诊断入口**：两种算法在 100k steps 后均未展示有效学习，问题可能不在算法选择，而在：
  1. 100k steps 训练量严重不足（4v4 环境复杂度高）
  2. 奖励信号稀疏或梯度不稳定
  3. 环境/观测/动作编码可能存在未发现的 bug
- **结论**：不继续完整矩阵（6 run），需要先诊断训练信号问题。
- **下一步建议**：
  1. 增加训练步数至 300k-500k，观察是否收敛
  2. 检查训练曲线（entropy、clip_fraction、value_loss）判断是否在学习
  3. 简化环境（2v2、更小地图）验证算法正确性
  4. 或使用 GPU 加速训练以获得更长训练量
