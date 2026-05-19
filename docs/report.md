# Execution Report

## STATUS: RL_MAINLINE_RESTORED

> 上次更新：2026-05-18 | plan.md 版本：RL 主线

## Last Execution

- 摘要：将项目叙事从 AgentBench 主线恢复为 MOBA/RL 主线。AgentBench 降级为应用层子系统，代码与测试完整保留。

## RL 主线当前状态

### 已完成

- [x] MiniMOBA 环境 + PettingZoo API
- [x] PPO / DualClipPPO 训练正确性修复
- [x] 324-way action mask + joint categorical policy
- [x] tower / base objective system
- [x] MAPPO / QMIX / COMA 算法框架
- [x] Self-play pool + Curriculum manager
- [x] LLM Planner MVP（RulePlanner + DummyLLMClient）
- [x] train / evaluate / run_ablation CLI
- [x] Evaluator + checkpoint utilities

### 核心未解决

- [ ] **ISSUE-F13**：objective reward shaping 提升了 tower_damage，但 hard_win_rate=0.0、base_exposed_rate=0.0、avg_base_damage=0.0

### Smoke 实验记录

当前 smoke 实验仅验证流水线可运行，不代表算法结论：

```bash
python -m hybrid_arena.scripts.train --algo ppo --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 3 --seed 42
```

详细 smoke 数据见 `docs/experiment-report-v0.md`。

## AgentBench 应用层验证

| scenario | total | metrics |
|---|---:|---|
| jd_resume_match | 2 | skill_recall=1.0, evidence_coverage=1.0 |
| telecom_rag | 3 | recall_at_k=1.0, citation_coverage=1.0, unsupported_answer_rate=0.0 |
| ticket_triage | 5 | accuracy=1.0, macro_f1=1.0, unknown_rate=0.0 |

## Full Verification

- `python -m compileall hybrid_arena`：通过
- `pytest hybrid_arena/ -v`：180 passed, 1 skipped
- `ruff check hybrid_arena`：All checks passed
