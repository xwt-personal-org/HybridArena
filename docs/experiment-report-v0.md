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

---

## sanity_2v2：简化环境训练信号诊断

> **目的**：验证训练闭环是否能在低复杂度环境下有效学习，区分「环境复杂度问题」与「训练闭环 bug」。
>
> **结论**：2v2 环境下 PPO 有明显学习信号（win_rate 33.3% vs 4v4 的 16.7%），但未达 50% 验收阈值。
> 不上 300k-500k 长训，需先修复 evaluator 和 reward 结构问题。

### 实验配置

- 配置文件：`configs/experiments/sanity_2v2.yaml`
- 训练步数：100,000
- 评估 episodes：30
- max_steps：200
- 训练环境：map_size=16, team_size=2, num_envs=4
- 训练耗时：2069s (~34.5 分钟, CPU, ~48 FPS)

### 结果

| algo | seed | opponent | win_rate | draw_rate | avg_reward | avg_len | avg_towers_destroyed | avg_tower_hp_advantage | fps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 42 | random | 0.333 | 0.333 | 6.183 | 200.0 | 0.6 | 120.0 | 369.6 |

### 训练曲线关键指标

| Step | Reward | EpLen | PolicyLoss | ValueLoss | Entropy | Clip% | KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | +6.69 | 200 | -0.008 | 0.896 | 4.546 | 6.7% | 0.006 |
| 10,240 | +13.27 | 180 | -0.014 | 1.222 | 4.238 | 15.4% | 0.011 |
| 20,224 | +5.09 | 43 | -0.026 | 1.227 | 4.194 | 17.3% | 0.011 |
| 50,176 | +5.27 | 40 | -0.025 | 0.996 | 4.169 | 18.2% | 0.012 |
| 100,000 | +6.20 | 35 | -0.034 | 1.405 | 4.237 | 13.5% | 0.009 |

### 分析

**正面信号（证明训练在学习）**：
- episode length 从 200（timeout）收敛到 34-40 步，说明 agent 学会了交战
- win_rate 从 4v4 的 16.7% 提升到 33.3%
- entropy 从 4.55 降到 ~4.0，policy 在变确定性
- KL 保持健康（0.006-0.014），policy 变化平稳
- avg_towers_destroyed = 0.6，agent 学会了推塔

**问题信号（未达 50% 阈值的原因）**：
1. **draw_rate = 33.3%**：三分之一对局 timeout（200 步），agent 没学会终结比赛
2. **evaluator 奖励跨两队平均**：`sum(rewards.values()) / len(rewards)` 包含了对方负奖励，稀释了胜负信号
3. **avg_reward 在训练中下降**：从 +9.71（早期）降到 +5.86（晚期），因 episode 缩短导致累积奖励减少
4. **value_loss 持续偏高**（0.9-2.1），value function 未充分收敛

### 根因判断

问题根源是**训练闭环结构性问题**，不是单纯的环境复杂度：

1. **evaluator 设计缺陷**：evaluator 对 reward 取两队平均，导致 win_rate 与 avg_reward 不一致
2. **reward 信号强度不足**：win=+5.0, lose=-5.0 对 200 步 episode 的占比偏低
3. **33% timeout**：agent 学会了交战但没学会推基地，需要更强的 objective 引导

### 下一步建议（ISSUE-F11）

1. 修复 evaluator：只计算 red team 的 reward（排除 blue team）
2. 检查 reward 基础设施：确认 win/lose reward 在 episode 结束时正确发放
3. 考虑增大 win/lose reward 或减小 time_penalty 权重
4. 修复后再跑 sanity_2v2 验证，目标 win_rate ≥ 50%
