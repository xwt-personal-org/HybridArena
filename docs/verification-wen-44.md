# WEN-44 scripted objective policy smoke 验证

## 目标

验证 MiniMOBA objective policy 与 base objective 可达性，记录关键指标并输出结论，用于 ISSUE-F13 后续判断。

## 验证命令

```bash
python -m hybrid_arena.scripts.objective_policy_smoke --episodes 3 --seed 42 --map-size 16 --team-size 2 --max-steps 500 --output results/wen-44-objective-policy-smoke.json
```

## 结果

```json
{
  "episodes": 3,
  "seed": 42,
  "map_size": 16,
  "team_size": 2,
  "max_steps": 500,
  "hard_win_rate": 1.0,
  "base_exposed_rate": 1.0,
  "avg_base_damage": 2000.0,
  "tower_damage": 2400.0,
  "avg_reward_margin": 29.39999999999999,
  "avg_length": 163.0,
  "terminal_reasons": ["base_destroyed", "base_destroyed", "base_destroyed"],
  "conclusion": "通过"
}
```

## 结论

- scripted objective policy 可以稳定摧毁蓝方两座塔并打掉基地。
- `hard_win_rate`、`base_exposed_rate`、`avg_base_damage`、`tower_damage` 均有正向信号。
- 本 smoke 只证明环境 objective/base 路径可达，不证明 RL 已学会 hard win；若训练仍出现 `hard_win_rate=0.0`、`base_exposed_rate=0.0`、`avg_base_damage=0.0`，优先排查 reward 尺度与训练设置。

## 环境备注

本地默认解释器是 Python 3.13，仓库声明支持 Python 3.10-3.12；为运行 smoke 单独安装了 `pettingzoo` / `pygame`。`supersuit` 因本机缺少 MSVC Build Tools 未安装，本次 smoke 未调用 `raw_env`。
