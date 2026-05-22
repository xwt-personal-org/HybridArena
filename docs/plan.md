# HybridArena 开发计划

生成日期：2026-05-18

## 元信息

- 项目：HybridArena
- 主线：LLM Planner × DRL Control · MiniMOBA 4v4 多智能体研究平台
- 应用层：AgentBench（JD 解析 / 通信 RAG / 工单分诊）已完成首版，保留为可选子系统
- 技术栈：Python 3.10+、PettingZoo、Gymnasium、NumPy、PyTorch、Pygame、Pytest、Ruff；LLM 可选 Transformers / TRL
- 状态：RL 工程闭环已搭好，正式实验与训练有效性验证为当前重点

## 总体优先级

| 优先级 | 目标 | 说明 |
|---|---|---|
| P0 | 代码真实性与 CI 加固 | 完整验证矩阵、CI 覆盖范围扩展、文档口径统一 |
| P0 | 训练正确性 | action mask、PPO old values、buffer mask、dual-clip 指标（已修复） |
| P0 | 环境目标闭环 | 塔、基地、队伍经济、objective rewards（已完成框架） |
| P1 | 正式 RL Baseline 实验 | 正式配置、seed sweep、训练信号判定 |
| P1 | 实验可复现 | CLI、evaluator、checkpoint、ablation |
| P1 | 环境与评估语义审计 | win_rate 视角、objective event 统计 |
| P2 | Self-play / Curriculum 调优 | 对手池、ELO 门控 |
| P2 | LLM Planner 数据闭环 | planner trace dataset、GRPO 前置验收 |
| P3 | LLM Planner 正式训练 | QLoRA GRPO，仅在 RL baseline 稳定后 |

## 当前阶段：P0 — 代码真实性与正式实验准备

### P0.0：M0 范围冻结与双主线边界

M0 已冻结 HybridArena 的双主线维护边界：

- MiniMOBA/RL 主线归属 M1：环境、动作系统、算法、训练/评估、ISSUE-F13。
- AgentBench 应用层归属 M2：`core`、`scenarios`、API、CLI、reporting、demo。
- Demo/API/docs 口径风险归属 M3。
- 发布门禁、Linear 状态回读与最终交接归属 M4。

范围冻结记录见 `docs/scope-freeze-m0.md`。M0 不实现算法或场景代码，只冻结事实基线、非目标、成功指标、测试门禁和已知风险。

### P0.1：完整运行本地验证矩阵

```bash
python -m pip install -e ".[dev,rl]"
ruff check hybrid_arena
python -m compileall hybrid_arena
pytest hybrid_arena/ -v
python -m hybrid_arena.scripts.train --algo ppo --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 3 --seed 42 --output results/eval_smoke.json
python -m hybrid_arena.scripts.run_ablation --episodes 1 --max-steps 50
python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 50 --render-mode none
```

### P0.2：ISSUE-F13 — 解决 objective reward shaping 有效性

- 现象：tower_damage 提升，但 hard_win_rate / base_exposed_rate / avg_base_damage 仍为 0
- 方向：先用 scripted objective policy 验证 base 可达性，再决定是否进入 300k-500k 长训
- Linear：WEN-44 单独跟踪，归属 M1 MiniMOBA/RL 稳定化
- 详见 `docs/issues.md`

### P0.3：文档与版本口径统一

- pyproject.toml 版本与 CHANGELOG 对齐
- README 实验表从 smoke 替换为正式结果（待 P1 完成）

## 下一阶段：P1 — 正式 RL Baseline 实验

- 新增正式实验配置（`configs/experiments/phase_b_baseline.yaml`）
- 让 run_ablation 支持配置文件驱动
- 运行正式 baseline 训练与评估
- 训练有效性判定脚本

## 远期：P2/P3 — LLM Planner 数据闭环与训练

- planner trace schema + JSONL recorder
- play_planner 支持 trace 输出
- GRPO 前置数据验收
- QLoRA GRPO 正式训练（RTX 4060 8GB，1.5B 级模型）

## AgentBench 应用层（已完成，保留维护）

三个业务场景（JD 解析、通信 RAG、工单分诊）+ FastAPI + CLI + Streamlit demo 已完成首版。
不再作为活跃开发主线，但保留代码与测试，不删除。
详见 `docs/.archive/plan-agentbench-v3.md`。

## 详细步骤参考

完整步骤分解见 `docs/hybridarena-next-step-plan.md`。
