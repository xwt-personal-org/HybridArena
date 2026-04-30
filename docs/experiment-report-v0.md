# HybridArena 实验报告 v0

## 环境设置

- Python: `3.10 <= version < 3.13`
- 安装: `pip install -e ".[dev,rl]"`
- 基础验证: `python -m compileall hybrid_arena`、`pytest hybrid_arena/minimoba/tests -v`

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

## 后续实验矩阵

| algo | seeds | opponents | metrics |
|---|---|---|---|
| ppo | 42, 123, 456 | random, rule_based | win_rate, avg_reward, avg_len, fps |
| ppo_dualclip | 42, 123, 456 | random, rule_based | win_rate, avg_reward, avg_len, fps |
