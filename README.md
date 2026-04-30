# HybridArena

**LLM-Decision x DRL-Control: A Hybrid Agent Training Platform for Imperfect-Information Strategy Games**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> 在自研的4v4不完全信息MOBA环境中，用LLM做高层战略决策，用DRL做低层微操控制，并用GRPO对LLM planner进行多回合强化学习微调。

## 亮点

- **自研 MiniMOBA-4v4 环境**：PettingZoo Parallel API，不完全信息，3种英雄，支持实时渲染
- **5种DRL算法统一对比**：PPO / Dual-clip PPO / MAPPO / QMIX / COMA
- **Self-Play + ELO + Curriculum Learning**：完整的对手池管理和自适应难度调度
- **LLM高层规划器**：StateTranslator + 策略桥接 + mock/API/local 三种推理模式
- **QLoRA GRPO 训练**：在消费级GPU上微调1.5B LLM战术规划器
- **严谨消融实验框架**：Evaluator + W&B logger + 多seed实验配置

## 快速开始

```bash
# 安装
pip install -e ".[dev,rl]"

# 运行环境测试
pytest hybrid_arena/minimoba/tests/ -v

# 运行算法测试
pytest hybrid_arena/algorithms/ -v

# PPO 训练
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --device cpu

# 评估
python -m hybrid_arena.scripts.evaluate --checkpoint checkpoints/...pt --opponent rule_based --episodes 50
```

## 架构

```
hybrid_arena/
├── minimoba/          # 4v4 MOBA 环境 (PettingZoo)
│   ├── env.py
│   ├── game_engine.py
│   ├── hero.py
│   ├── map_generator.py
│   ├── reward_shaper.py
│   ├── renderer.py
│   ├── wrappers.py
│   └── tests/
├── algorithms/        # 5种DRL算法
│   ├── networks.py    # 共享网络 (MapEncoder + StateEncoder + ActorCritic)
│   ├── ppo/           # PPO + DualClipPPO
│   ├── mappo/         # MAPPO (CTDE centralized critic)
│   ├── qmix/          # QMIX (value decomposition + mixing network)
│   ├── coma/          # COMA (counterfactual baseline)
│   └── self_play/     # Self-Play Manager + ELO + Curriculum
├── training/
│   ├── trainer.py     # PPO/MAPPO 训练循环
│   ├── buffer.py      # Rollout Buffer (GAE)
│   ├── evaluator.py   # 对战胜率 / ELO / KDA 评估
│   ├── logger.py      # W&B 日志封装
│   └── grpo_trainer.py  # QLoRA GRPO (LLM fine-tuning)
├── inference/         # LLM 规划器
│   ├── state_translator.py
│   ├── llm_planner.py
│   └── strategy_bridge.py
├── demo/              # Streamlit Demo
└── configs/           # YAML 配置
```

## 动作与观测空间

- **动作空间**: `MultiDiscrete([9, 4, 9])` = 移动(8方向+不动) × 技能(普攻/技能1/技能2/不攻击) × 目标(8敌方+野怪+不选) = **324** 种组合
- **joint action mask**: 环境输出 `action_mask` 为 `(324,)`，语义是完整的 move × skill × target 合法组合掩码。
- **Policy mask 处理**: ActorCritic 内部保留 move/skill/target 三个 head，通过广播相加生成 324-way joint logits，再对完整 joint action mask 做 masking；不会把 `(324,)` mask 切片成 9/4/9 三个独立 mask。
- **观测空间**: Dict
  - `local_map`: (11, 11, 11) 局部视野
  - `self_state`: (20,) 自身状态
  - `teammate_states`: (3, 15) 队友状态
  - `global_info`: (10,) 全局信息
  - `action_mask`: (324,) 合法动作掩码

## Objective 与奖励

- **目标物**: 地图含 red/blue tower 与 base，塔被摧毁会更新队伍经济、`red_towers` / `blue_towers` 和 objective reward；base 只能在对应队伍 tower 全灭后被普攻伤害，base 摧毁后结束对局。
- **奖励项**: `kill` / `death` / `assist` / `damage` / `heal` / `tower` / `tower_lost` / `base` / `win` / `lose` / `time_penalty`。
- **结构物伤害限制**: 当前版本只有普攻能伤害 tower/base，技能不伤害结构物。

## 算法对比

| 算法 | 核心特点 | 状态 |
|------|---------|------|
| PPO (vanilla) | Clipped surrogate + GAE | 已实现 |
| Dual-clip PPO | 上下界双裁剪 + 自适应熵衰减 | 已实现 |
| MAPPO | CTDE 中心化critic | 已实现 |
| QMIX | 单调性价值分解 (mixing network) | 已实现 |
| COMA | Counterfactual baseline | 已实现 (简化版) |

## 实验配置

```yaml
# configs/default.yaml
environment:
  map_size: 32
  team_size: 4
  max_steps: 1000

training:
  total_timesteps: 3_000_000
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95

self_play:
  pool_size: 10
  win_threshold: 0.55

curriculum:
  levels: [entry, basic, standard, advanced]
```

## 技术栈

- Python 3.10+, PyTorch 2.x
- PettingZoo + Gymnasium
- QLoRA (peft + transformers + bitsandbytes)
- W&B (实验追踪)
- Streamlit (Demo)

## 测试

```bash
# 全部测试
pytest hybrid_arena/ -v

# 按模块
pytest hybrid_arena/minimoba/tests/ -v
pytest hybrid_arena/algorithms/self_play/tests/ -v
pytest hybrid_arena/algorithms/mappo/tests/ -v
pytest hybrid_arena/algorithms/qmix/tests/ -v
pytest hybrid_arena/algorithms/coma/tests/ -v
pytest hybrid_arena/inference/tests/ -v
pytest hybrid_arena/training/tests/ -v
```

## 开发状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase A | 环境 + Rule-Based Baseline | 完成 |
| Phase B | 5种DRL算法 + Self-Play + Evaluator | 完成 |
| Phase C | LLM Planner (StateTranslator + LangGraph + Bridge) | 完成 |
| Phase D | QLoRA GRPO 训练管线 | 完成 |
| Phase E | Streamlit Demo + 文档 | 完成 |

## 当前实现状态

| 模块 | 状态 |
|------|------|
| 项目基线与开发护栏 | 已完成 |
| joint action mask 与 ActorCritic 324-way policy | 已完成 |
| PPO / DualClipPPO old values、action masks、value clipping | 已完成 |
| tower/base objective game | 已完成 |
| checkpoint、Evaluator、train/evaluate/ablation CLI | 已完成 |
| `num_envs` 同步多环境 runner、SelfPlayPool、CurriculumManager | 已完成 |
| RulePlanner / Dummy LLMPlanner / MacroActionAdapter | 已完成 |

## 训练与评估命令

```bash
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --device cpu
python -m hybrid_arena.scripts.evaluate --checkpoint checkpoints/...pt --opponent rule_based --episodes 50
python -m hybrid_arena.scripts.run_ablation --episodes 1 --max-steps 50
python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 50 --render-mode none
```

## 实验结果表

| algo | seed | opponent | win_rate | avg_reward | avg_len | fps |
|---|---:|---|---:|---:|---:|---:|
| ppo | 42 | random | 待正式实验 | 待正式实验 | 待正式实验 | 待正式实验 |
| ppo_dualclip | 42 | rule_based | 待正式实验 | 待正式实验 | 待正式实验 | 待正式实验 |

## Known Limitations

- 当前 ablation 默认 smoke 参数用于验证流水线，不代表正式算法结论。
- LLM Planner MVP 默认使用 DummyLLMClient 或 RulePlanner，测试不调用外部 API。
- GRPO/QLoRA 不在本阶段实现，需在 RL baseline 稳定后再接入。

## License

MIT
