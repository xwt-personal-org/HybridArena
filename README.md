# HybridArena

LLM × DRL 混合智能体训练与基准平台——在不完全信息策略对战环境中，用 LLM 做高层战略决策，用 DRL 做低层微操控制。

## 项目定位

一个开源项目同时命中三个赛道：
- **游戏 AI**（腾讯/网易/米哈游）：自定义 PettingZoo 环境 + dual-clip PPO + self-play
- **通用 RL 算法工程师**：PPO/MAPPO/QMIX/COMA 统一对比 + 消融实验
- **AI Agent 开发**：LangGraph + CrewAI + ReAct/Reflexion + GRPO 微调 LLM

## 开发状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| A | MiniMOBA 4v4 环境 + Rule-based baseline | ✅ 完成 |
| B | PPO / Dual-clip PPO + 训练回路 | ✅ 完成 |
| C | LLM 高层规划器 (LangGraph + CrewAI) | ⏳ 计划中 |
| D | QLoRA GRPO 微调 LLM planner | ⏳ 计划中 |

## 快速开始

### 安装

```bash
# Python 3.10+
pip install -e .           # 环境 + agents
pip install -e ".[rl]"     # + PyTorch (DRL 训练)
pip install -e ".[dev]"    # + 测试/lint 工具
pip install -e ".[all]"    # 全部依赖
```

### 运行测试

```bash
pytest hybrid_arena/minimoba/tests/ -v
```

### 环境基准

```bash
python hybrid_arena/scripts/benchmark_fps.py
```

### DRL 训练

```python
from hybrid_arena.algorithms.ppo.config import PPOConfig
from hybrid_arena.training.trainer import Trainer

config = PPOConfig(total_timesteps=1_000_000)
trainer = Trainer(config, algo_type="ppo_dualclip")
trainer.train()
```

## 架构

```
hybrid_arena/
├── minimoba/          # PettingZoo 4v4 MOBA 环境
│   ├── env.py         #   MiniMOBAEnv (ParallelEnv)
│   ├── game_engine.py #   GameState (战争迷雾+战斗+同步决策)
│   ├── hero.py        #   HeroConfig + HERO_POOL + HeroState
│   ├── map_generator.py  #  地图生成
│   ├── reward_shaper.py  #  奖励配置
│   ├── renderer.py    #   Pygame 2D 渲染
│   ├── wrappers.py    #   SingleAgent wrapper
│   ├── agents/        #   RandomAgent, RuleBasedAgent
│   └── tests/         #   API 合规 + 环境 + 奖励测试
├── algorithms/        # 算法实现
│   ├── networks.py    #   MapEncoder + StateEncoder + ActorCritic (164K)
│   └── ppo/           #   PPO + DualClipPPO + config
├── training/          # 训练基础设施
│   ├── buffer.py      #   RolloutBuffer + GAE
│   └── trainer.py     #   训练循环
├── configs/           # YAML 配置文件
└── scripts/           # play_human.py, benchmark_fps.py
```

## 动作空间

`MultiDiscrete([9, 4, 9])` = 移动(9) × 技能(4) × 目标(9) = 324

## 观测空间

`Dict{local_map(11,11,11), self_state(20), teammate_states(3,15), global_info(10), action_mask(324)}`

## 技术栈

Python 3.10+ · PyTorch 2.x · PettingZoo · Gymnasium · CleanRL-style PPO · Ruff · Pytest

## 硬件约束

目标硬件：RTX 4060 Laptop (8GB VRAM)。当前开发机器：MX450 (2GB)，足以运行环境和 DRL 训练，LLM 推理需切换至 RTX 4060。

详见 `docs/` 中的完整方案文档。
