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
