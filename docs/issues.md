# HybridArena 问题记录

> Codex 在执行中遇到 `docs/plan.md` 未覆盖的情况时记录于此。

## 当前执行阻塞

（暂无）

## 已知但已纳入计划的问题

- action mask 语义不一致：已纳入模块 1、模块 2。
- PPO clipped value loss 无效：已纳入模块 2。
- DualClipPPO 指标错误：已纳入模块 2。
- 塔、基地、队伍经济未闭环：已纳入模块 3。
- `num_envs` 未使用：已纳入模块 5。
- 训练 CLI、评估器、checkpoint 缺失：已纳入模块 4。
- LLM Planner 目录为空：已纳入模块 6。

## 新发现问题

### ISSUE-F0：env.close() 在无 pygame 环境下崩溃
- 严重级别：P1
- 影响：无 pygame 安装时 smoke test 在 close() 调用失败
- 复现：`pytest hybrid_arena/minimoba/tests/test_smoke.py -v`（不装 pygame）
- 修复状态：已修复（Phase F0.3），`env.close()` 中 pygame 改为 try/except ImportError
