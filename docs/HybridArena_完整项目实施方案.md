# HybridArena 完整项目实施方案

## —— 从零到简历级成果的全流程指南

---

## 〇、项目定位与核心逻辑

### 为什么是这个项目

当前市场上99%的RL求职者简历长这样：「用PPO跑了CartPole」「在Atari上复现了DQN」「用MAPPO做了星际微操」。这些项目的问题不是"错"，而是**没有差异化信号**——面试官看不出你比其他候选人强在哪里。

HybridArena的设计逻辑是：**用一个项目同时命中三个赛道，并在每个赛道上都展现出超越"调包跑通"的深度**。

| 目标赛道 | 本项目命中的技术点 |
|---|---|
| 游戏AI（腾讯/网易/米哈游） | 自定义游戏环境 + dual-clip PPO + self-play + curriculum learning + reward shaping + 可视化Demo |
| 通用RL算法工程师 | PPO/MAPPO/QMIX/MuZero 五算法统一对比 + 严谨消融实验 + 分布式训练 |
| AI Agent开发 | LangGraph状态机 + CrewAI角色化 + ReAct/Reflexion + GRPO微调LLM planner |

### 一句话定义

> **HybridArena**：一个开源的不完全信息策略对战平台，在自研的4v4小队对战环境中，用LLM做高层战略决策（何时开团、分推、撤退），用DRL做低层微操控制（走位、技能释放、目标选择），并用GRPO对LLM planner进行多回合强化学习微调，实现"能思考的AI队伍"。

### 项目最终产出清单

完成后你将拥有以下可展示成果：

1. **GitHub仓库**（工业级README + 架构图 + 一键训练脚本）
2. **在线Demo**（Hugging Face Spaces / Streamlit，面试官点击即玩）
3. **W&B公开报告**（所有实验曲线、消融表格一键查看）
4. **2-3篇技术博客**（知乎/Medium，展示"自己的思考"）
5. **简历项目描述**（精确到数字的成果表述）

---

## 一、整体架构设计

### 1.1 系统分层架构

```
┌─────────────────────────────────────────────────────┐
│                   HybridArena 系统架构                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │        Layer 3: Agentic RL Training           │  │
│  │  ┌─────────────┐  ┌──────────────────────┐   │  │
│  │  │ GRPO Trainer │  │ Self-Play Manager    │   │  │
│  │  │ (vLLM/TRL)  │  │ (ELO Rating System)  │   │  │
│  │  └─────────────┘  └──────────────────────┘   │  │
│  └───────────────────────────────────────────────┘  │
│                         │                           │
│  ┌───────────────────────────────────────────────┐  │
│  │        Layer 2: High-Level LLM Planner        │  │
│  │  ┌──────────┐  ┌───────────┐  ┌───────────┐  │  │
│  │  │ LangGraph│  │ CrewAI    │  │ Reflexion  │  │  │
│  │  │ 状态机    │  │ NPC角色化 │  │ 自我反思   │  │  │
│  │  └──────────┘  └───────────┘  └───────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│                         │                           │
│  ┌───────────────────────────────────────────────┐  │
│  │        Layer 1: Low-Level DRL Controller      │  │
│  │  ┌──────┐ ┌──────────┐ ┌──────┐ ┌─────────┐ │  │
│  │  │ PPO  │ │dual-clip │ │MAPPO │ │  QMIX   │ │  │
│  │  │      │ │  PPO     │ │      │ │         │ │  │
│  │  └──────┘ └──────────┘ └──────┘ └─────────┘ │  │
│  └───────────────────────────────────────────────┘  │
│                         │                           │
│  ┌───────────────────────────────────────────────┐  │
│  │        Layer 0: Game Environment              │  │
│  │  ┌──────────────────────────────────────────┐ │  │
│  │  │   MiniMOBA-4v4 (PettingZoo Parallel API) │ │  │
│  │  │   不完全信息 · 连续+离散混合动作空间      │ │  │
│  │  └──────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 1.2 环境设计：MiniMOBA-4v4

**为什么自研环境而不用现成的？** 因为自研环境本身就是面试加分项——它证明你理解"环境-Agent接口设计"这个RL工程的核心难题。腾讯绝悟、网易伏羲都要求候选人"能独立设计训练环境"。

**环境规格：**

```
名称：MiniMOBA-4v4
类型：2D俯视角简化MOBA
地图：32×32网格（含草丛/障碍物/野怪/基地）
队伍：红方4人 vs 蓝方4人
英雄：3种角色（坦克/输出/辅助），各有2个主动技能+1个被动
视野：每个英雄只能看到周围5格（不完全信息的核心来源）
回合制/实时：离散时间步，每步所有agent同时决策

观测空间（每个agent）：
  - 局部地图: (11, 11, C) — 以自身为中心的11×11局部视野
    - C通道: 地形(1) + 己方单位(4) + 敌方可见单位(4) + 野怪(1) + 建筑(1) = 11通道
  - 自身状态: (20,) — HP/MP/等级/技能CD/位置/朝向/buff状态
  - 队友状态: (3, 15) — 队友的HP/位置/技能CD（通过通信共享）
  - 全局信息: (10,) — 时间/双方击杀数/经济差/塔存活状态

动作空间（离散，MultiDiscrete）：
  - 移动方向: Discrete(9)  — 8方向+不动
  - 技能选择: Discrete(4)  — 普攻/技能1/技能2/不攻击
  - 目标选择: Discrete(9)  — 8个敌方+野怪目标+不选择
  - 总动作空间大小: 9 × 4 × 9 = 324

奖励设计（多目标，可配置权重）：
  - r_kill = +1.0 （击杀敌方英雄）
  - r_death = -0.8 （自身死亡）
  - r_assist = +0.3 （助攻）
  - r_tower = +2.0 （推塔）
  - r_tower_lost = -2.0 （己方塔被推）
  - r_farm = +0.1 （击杀野怪/小兵获取经济）
  - r_damage = +0.01 × damage_dealt （造成伤害）
  - r_heal = +0.01 × heal_amount （治疗队友）
  - r_win = +5.0 / r_lose = -5.0 （最终胜负）
  - r_time = -0.001 × step （时间惩罚，鼓励积极进攻）

单局长度：最多2000步（约5分钟游戏时间）
终止条件：一方基地被摧毁 或 达到最大步数（按剩余血量判胜负）
```

### 1.3 API设计

```python
# 环境遵循 PettingZoo Parallel API
# 同时提供 Gymnasium single-agent wrapper（用于单英雄训练）

import minimoba

# === PettingZoo 多智能体模式 ===
env = minimoba.parallel_env(
    map_size=32,
    team_size=4,
    hero_pool=["tank", "dps", "support"],
    reward_config={
        "kill": 1.0, "death": -0.8, "tower": 2.0,
        "farm": 0.1, "win": 5.0, "time_penalty": -0.001
    },
    fog_of_war=True,          # 不完全信息
    render_mode="rgb_array",  # 支持录像
    max_steps=2000,
)

obs, infos = env.reset(seed=42)
# obs = {"red_0": {...}, "red_1": {...}, ..., "blue_3": {...}}

while env.agents:
    actions = {agent: policy(obs[agent]) for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

# === Gymnasium 单智能体 wrapper ===
single_env = minimoba.make_single_agent(
    team_size=4,
    control_agent="red_0",          # 你控制红方0号
    teammate_policy="pretrained",   # 队友用预训练策略
    opponent_policy="rule_based",   # 对手用规则AI
)
```

### 1.4 技术栈全景

```
语言：           Python 3.10+ / TypeScript (可视化前端)
深度学习框架：    PyTorch 2.x
RL框架：         CleanRL (自定义fork) + 自实现算法
多智能体：       PettingZoo 1.24+ / SuperSuit
环境渲染：       Pygame / Matplotlib
LLM推理：        vLLM / Transformers
LLM Agent框架：  LangGraph 0.2+ / CrewAI 0.x
LLM训练：        TRL (GRPO Trainer) / OpenRLHF
模型：           Qwen2.5-7B-Instruct (planner) / Qwen2.5-1.5B (轻量实验)
实验追踪：       Weights & Biases
可视化：         Streamlit / Gradio
代码质量：       pytest / pre-commit / ruff
容器化：         Docker + docker-compose
版本管理：       Git + GitHub Actions CI
```

---

## 二、分阶段实施方案

### ═══════════════════════════════════════════
### 阶段 A：环境搭建与Rule-Based Baseline
### 预计时间：3-4周
### ═══════════════════════════════════════════

#### A.1 前置学习（第1周前半）

**必读材料：**

- PettingZoo 官方文档：Parallel API 部分（https://pettingzoo.farama.org/）
- Gymnasium 文档：自定义环境创建（https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/）
- 王者绝悟环境设计论文的环境描述部分（理解MOBA环境的观测/动作空间设计）

**动手练习：**

```python
# 练习1：跑通 PettingZoo 自带的 simple_spread_v3（MARL入门环境）
from pettingzoo.mpe import simple_spread_v3
env = simple_spread_v3.env(N=3, max_cycles=25)
env.reset()
for agent in env.agent_iter():
    obs, rew, term, trunc, info = env.last()
    action = env.action_space(agent).sample()
    env.step(action)

# 练习2：用 Parallel API 重写上面的交互循环
from pettingzoo.mpe import simple_spread_v3
env = simple_spread_v3.parallel_env(N=3, max_cycles=25)
obs, infos = env.reset()
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rews, terms, truncs, infos = env.step(actions)
```

#### A.2 环境核心实现（第1-2周）

**目录结构：**

```
hybrid_arena/
├── minimoba/
│   ├── __init__.py
│   ├── env.py              # 核心环境逻辑
│   ├── game_engine.py      # 游戏引擎（地图、碰撞、伤害计算）
│   ├── hero.py             # 英雄类定义（属性、技能）
│   ├── map_generator.py    # 地图生成（含草丛、野怪刷新点）
│   ├── reward_shaper.py    # 可配置奖励函数
│   ├── renderer.py         # Pygame渲染
│   ├── wrappers.py         # 单智能体wrapper、观测标准化wrapper
│   └── tests/
│       ├── test_env.py     # 环境正确性测试
│       ├── test_api.py     # PettingZoo API合规测试
│       └── test_reward.py  # 奖励函数单元测试
├── agents/
│   ├── rule_based.py       # Rule-based baseline（有限状态机）
│   └── random_agent.py     # 随机策略
├── configs/
│   └── default.yaml        # 环境默认配置
├── scripts/
│   ├── play_human.py       # 人类可玩的调试模式
│   └── benchmark_fps.py    # 环境性能基准测试
└── README.md
```

**游戏引擎核心类（game_engine.py）：**

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class HeroConfig:
    """英雄模板配置"""
    name: str
    role: str                    # "tank" / "dps" / "support"
    max_hp: float
    max_mp: float
    attack_damage: float
    attack_range: int            # 格子数
    move_speed: int              # 每步移动格子数
    skill_1: dict                # {"damage": x, "range": y, "cd": z, "mp_cost": w, "type": "aoe/single/heal"}
    skill_2: dict
    passive: dict                # {"type": "regen/shield/aura", "value": x}

# 预定义三种英雄
HERO_POOL = {
    "tank": HeroConfig(
        name="Guardian", role="tank",
        max_hp=1000, max_mp=300,
        attack_damage=30, attack_range=1, move_speed=1,
        skill_1={"damage": 80, "range": 2, "cd": 5, "mp_cost": 40, "type": "aoe", "effect": "stun_1"},
        skill_2={"damage": 0, "range": 0, "cd": 8, "mp_cost": 60, "type": "self", "effect": "shield_200"},
        passive={"type": "damage_reduction", "value": 0.15}
    ),
    "dps": HeroConfig(
        name="Striker", role="dps",
        max_hp=600, max_mp=400,
        attack_damage=60, attack_range=3, move_speed=2,
        skill_1={"damage": 150, "range": 4, "cd": 3, "mp_cost": 50, "type": "single"},
        skill_2={"damage": 200, "range": 5, "cd": 10, "mp_cost": 80, "type": "line", "effect": "slow_2"},
        passive={"type": "crit_chance", "value": 0.2}
    ),
    "support": HeroConfig(
        name="Sage", role="support",
        max_hp=500, max_mp=600,
        attack_damage=20, attack_range=2, move_speed=1,
        skill_1={"damage": 0, "range": 3, "cd": 4, "mp_cost": 50, "type": "heal", "heal": 120},
        skill_2={"damage": 60, "range": 4, "cd": 6, "mp_cost": 40, "type": "aoe", "effect": "vision_3"},
        passive={"type": "mp_regen", "value": 5}
    ),
}

class GameState:
    """游戏状态管理器"""
    def __init__(self, map_size: int = 32, team_size: int = 4):
        self.map_size = map_size
        self.team_size = team_size

        # 地图层
        self.terrain = np.zeros((map_size, map_size), dtype=np.int8)
        # 0=空地, 1=墙壁, 2=草丛, 3=红方基地, 4=蓝方基地, 5=红方塔, 6=蓝方塔

        # 英雄状态
        self.heroes: Dict[str, HeroState] = {}  # "red_0" -> HeroState

        # 野怪状态
        self.jungle_camps: List[JungleCamp] = []

        # 全局状态
        self.step_count = 0
        self.red_kills = 0
        self.blue_kills = 0
        self.red_gold = 0
        self.blue_gold = 0

    def get_observation(self, agent_id: str) -> dict:
        """获取单个agent的局部观测（不完全信息的核心）"""
        hero = self.heroes[agent_id]
        team = "red" if agent_id.startswith("red") else "blue"

        # 局部地图（以自身为中心的11×11）
        local_map = self._extract_local_map(hero.x, hero.y, radius=5)

        # 战争迷雾：只能看到视野内的敌方单位
        visible_enemies = self._get_visible_enemies(hero, team)

        # 编码局部地图为多通道tensor
        map_channels = self._encode_map_channels(
            local_map, hero, visible_enemies, team
        )  # shape: (11, 11, 11)

        # 自身状态向量
        self_state = np.array([
            hero.hp / hero.max_hp,
            hero.mp / hero.max_mp,
            hero.level / 15.0,
            hero.skill_1_cd / hero.skill_1_max_cd,
            hero.skill_2_cd / hero.skill_2_max_cd,
            hero.x / self.map_size,
            hero.y / self.map_size,
            hero.attack_damage / 200.0,
            hero.gold / 5000.0,
            hero.exp / 1000.0,
            # ... 其余归一化特征
        ], dtype=np.float32)  # shape: (20,)

        # 队友状态（通过通信共享，始终可见）
        teammate_states = self._get_teammate_states(agent_id, team)  # shape: (3, 15)

        # 全局信息
        global_info = np.array([
            self.step_count / 2000.0,
            self.red_kills / 50.0,
            self.blue_kills / 50.0,
            (self.red_gold - self.blue_gold) / 10000.0,
            # ... 塔存活状态等
        ], dtype=np.float32)  # shape: (10,)

        return {
            "local_map": map_channels,       # (11, 11, 11)
            "self_state": self_state,         # (20,)
            "teammate_states": teammate_states,  # (3, 15)
            "global_info": global_info,       # (10,)
            "action_mask": self._get_action_mask(hero),  # (324,) 合法动作掩码
        }

    def _get_visible_enemies(self, hero, team: str) -> list:
        """战争迷雾实现：只返回视野范围内的敌方英雄"""
        enemy_team = "blue" if team == "red" else "red"
        visible = []
        for eid, ehero in self.heroes.items():
            if not eid.startswith(enemy_team):
                continue
            if ehero.is_dead:
                continue
            dist = abs(hero.x - ehero.x) + abs(hero.y - ehero.y)  # 曼哈顿距离
            # 草丛中的敌人需要更近才能看到
            vision_range = 5
            if self.terrain[ehero.y, ehero.x] == 2:  # 草丛
                vision_range = 2
            if dist <= vision_range:
                visible.append(ehero)
        return visible

    def step(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """执行一步，返回每个agent的奖励"""
        rewards = {agent: 0.0 for agent in self.heroes}

        # 1. 解析动作
        parsed_actions = {}
        for agent_id, action in actions.items():
            move_dir = action[0]      # 0-8
            skill_choice = action[1]  # 0-3
            target_choice = action[2] # 0-8
            parsed_actions[agent_id] = (move_dir, skill_choice, target_choice)

        # 2. 同时执行移动（解决冲突）
        self._resolve_movements(parsed_actions)

        # 3. 同时执行技能/攻击（计算伤害）
        events = self._resolve_combat(parsed_actions)

        # 4. 计算奖励
        for event in events:
            self._apply_reward(event, rewards)

        # 5. 更新全局状态
        self._update_global_state()
        self.step_count += 1

        return rewards
```

**PettingZoo环境封装（env.py）：**

```python
import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from .game_engine import GameState, HERO_POOL

class MiniMOBAEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "minimoba_v1",
    }

    def __init__(
        self,
        map_size=32,
        team_size=4,
        hero_assignments=None,
        reward_config=None,
        fog_of_war=True,
        max_steps=2000,
        render_mode=None,
    ):
        super().__init__()
        self.map_size = map_size
        self.team_size = team_size
        self.fog_of_war = fog_of_war
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 默认英雄分配：每队1坦克+2输出+1辅助
        self.hero_assignments = hero_assignments or {
            "red_0": "tank", "red_1": "dps", "red_2": "dps", "red_3": "support",
            "blue_0": "tank", "blue_1": "dps", "blue_2": "dps", "blue_3": "support",
        }

        self.possible_agents = [f"{team}_{i}"
            for team in ["red", "blue"] for i in range(team_size)]
        self.agents = self.possible_agents[:]

        # 奖励配置
        self.reward_config = reward_config or {
            "kill": 1.0, "death": -0.8, "assist": 0.3,
            "tower": 2.0, "tower_lost": -2.0,
            "farm": 0.1, "damage": 0.01, "heal": 0.01,
            "win": 5.0, "lose": -5.0, "time_penalty": -0.001,
        }

        self.game_state = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({
            "local_map": spaces.Box(0, 1, shape=(11, 11, 11), dtype=np.float32),
            "self_state": spaces.Box(-1, 1, shape=(20,), dtype=np.float32),
            "teammate_states": spaces.Box(-1, 1, shape=(3, 15), dtype=np.float32),
            "global_info": spaces.Box(-1, 1, shape=(10,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(324),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.MultiDiscrete([9, 4, 9])  # 移动×技能×目标

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.game_state = GameState(self.map_size, self.team_size)
        self.game_state.initialize_map()
        self.game_state.spawn_heroes(self.hero_assignments)
        self.game_state.spawn_jungle_camps()

        observations = {
            agent: self.game_state.get_observation(agent)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # 执行游戏逻辑
        rewards = self.game_state.step(actions)

        # 添加时间惩罚
        for agent in rewards:
            rewards[agent] += self.reward_config["time_penalty"]

        # 检查终止条件
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        if self.game_state.is_game_over():
            winner = self.game_state.get_winner()
            for agent in self.agents:
                team = "red" if agent.startswith("red") else "blue"
                if team == winner:
                    rewards[agent] += self.reward_config["win"]
                else:
                    rewards[agent] += self.reward_config["lose"]
                terminations[agent] = True

        if self.game_state.step_count >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True

        # 获取新观测
        observations = {
            agent: self.game_state.get_observation(agent)
            for agent in self.agents
        }
        infos = {agent: {"episode_step": self.game_state.step_count}
                 for agent in self.agents}

        # 移除死亡/终止的agent
        self.agents = [a for a in self.agents
                       if not terminations[a] and not truncations[a]]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "rgb_array":
            return self.game_state.render_to_array()
        elif self.render_mode == "human":
            self.game_state.render_pygame()
```

**Rule-Based Baseline（agents/rule_based.py）：**

```python
"""
有限状态机 Rule-Based AI
状态：巡逻 → 发现敌人 → 交战 → 撤退/追击 → 回城
这是所有RL算法的对照基线
"""

class RuleBasedAgent:
    """简单的有限状态机AI，作为baseline"""

    def __init__(self):
        self.state = "patrol"

    def act(self, obs: dict) -> np.ndarray:
        hero_state = obs["self_state"]
        hp_ratio = hero_state[0]  # hp / max_hp

        # 状态转移
        if hp_ratio < 0.2:
            self.state = "retreat"
        elif self._detect_enemy(obs):
            if hp_ratio > 0.5:
                self.state = "engage"
            else:
                self.state = "retreat"
        else:
            self.state = "patrol"

        # 状态行为
        if self.state == "patrol":
            return self._patrol_action(obs)
        elif self.state == "engage":
            return self._engage_action(obs)
        elif self.state == "retreat":
            return self._retreat_action(obs)

    def _engage_action(self, obs):
        """交战逻辑：优先攻击最低血量的敌人"""
        enemies = self._get_visible_enemies(obs)
        if not enemies:
            return self._patrol_action(obs)

        # 找最低血量敌人
        target_idx = min(range(len(enemies)),
                        key=lambda i: enemies[i]["hp_ratio"])

        # 移动朝向目标
        move_dir = self._direction_to_target(obs, enemies[target_idx])

        # 优先释放技能（如果CD好了）
        skill = 0  # 普攻
        if obs["self_state"][3] == 0:  # 技能1 CD=0
            skill = 1
        elif obs["self_state"][4] == 0:  # 技能2 CD=0
            skill = 2

        return np.array([move_dir, skill, target_idx])
```

#### A.3 环境测试与质量保证（第2-3周）

**API合规测试（必须通过，否则后续所有RL训练都会出问题）：**

```python
# tests/test_api.py
from pettingzoo.test import parallel_api_test, seed_test
from minimoba import parallel_env

def test_parallel_api():
    """PettingZoo官方API合规测试"""
    env = parallel_env()
    parallel_api_test(env, num_cycles=100)

def test_seed_determinism():
    """确保相同seed产生相同轨迹"""
    env1 = parallel_env()
    env2 = parallel_env()

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    for key in obs1:
        for obs_key in obs1[key]:
            np.testing.assert_array_equal(obs1[key][obs_key], obs2[key][obs_key])

def test_action_mask_validity():
    """确保action_mask正确排除非法动作"""
    env = parallel_env()
    obs, _ = env.reset()
    for agent in env.agents:
        mask = obs[agent]["action_mask"]
        assert mask.sum() > 0, f"Agent {agent} has no valid actions"

def test_reward_range():
    """确保奖励在合理范围内"""
    env = parallel_env()
    obs, _ = env.reset()
    for _ in range(100):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        for agent, reward in rewards.items():
            assert -10 < reward < 10, f"Reward out of range: {reward}"

def test_fps_benchmark():
    """环境性能测试：目标 > 5000 steps/sec（8 envs并行）"""
    import time
    env = parallel_env()
    obs, _ = env.reset()
    start = time.time()
    steps = 0
    for _ in range(1000):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, terms, truncs, _ = env.step(actions)
        steps += 1
        if any(terms.values()) or any(truncs.values()):
            obs, _ = env.reset()
    elapsed = time.time() - start
    fps = steps / elapsed
    print(f"Environment FPS: {fps:.0f}")
    assert fps > 500, f"Environment too slow: {fps:.0f} FPS"
```

**Rule-Based Baseline指标（用于后续所有RL算法的对照）：**

```python
# scripts/evaluate_baseline.py
def evaluate_rule_based(n_games=200):
    """记录rule-based AI的胜率作为后续所有算法的底线"""
    env = parallel_env()
    red_agent = RuleBasedAgent()
    blue_agent = RuleBasedAgent()

    red_wins, blue_wins, draws = 0, 0, 0
    episode_lengths = []
    episode_rewards = []

    for game in range(n_games):
        obs, _ = env.reset(seed=game)
        total_reward = {a: 0 for a in env.possible_agents}
        step = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent.startswith("red"):
                    actions[agent] = red_agent.act(obs[agent])
                else:
                    actions[agent] = blue_agent.act(obs[agent])
            obs, rewards, terms, truncs, _ = env.step(actions)
            for a, r in rewards.items():
                total_reward[a] += r
            step += 1

        # 统计结果
        episode_lengths.append(step)
        episode_rewards.append(total_reward)
        winner = env.game_state.get_winner()
        if winner == "red": red_wins += 1
        elif winner == "blue": blue_wins += 1
        else: draws += 1

    print(f"Rule-Based Baseline (n={n_games}):")
    print(f"  Red wins: {red_wins/n_games:.1%}")
    print(f"  Draws: {draws/n_games:.1%}")
    print(f"  Avg episode length: {np.mean(episode_lengths):.0f}")
    print(f"  Avg episode reward (red): {np.mean([r['red_0'] for r in episode_rewards]):.2f}")
```

#### A.4 阶段A产出检查清单

- [ ] MiniMOBA-4v4 环境通过 `parallel_api_test`
- [ ] 环境 FPS > 500（单进程）
- [ ] Rule-Based baseline 双方胜率各约50%（平衡性验证）
- [ ] Pygame渲染正常，能录制对战回放
- [ ] `pip install -e .` 可安装，README含环境文档
- [ ] 提交到GitHub，第一个release tag `v0.1.0-env`


### ═══════════════════════════════════════════
### 阶段 B：DRL微观控制（5种算法统一对比）
### 预计时间：5-6周
### ═══════════════════════════════════════════

#### B.1 前置学习（与阶段A并行，持续2周）

**必读论文（按优先级排序）：**

1. **PPO**（Schulman 2017）—— 重点理解clip机制和GAE
2. **"The 37 Implementation Details of PPO"**（CleanRL博客）—— 这篇博客比任何教科书都重要
3. **王者绝悟1v1论文**（Ye et al., AAAI 2020）—— 理解dual-clip PPO
4. **MAPPO论文**（Yu et al., 2022, "The Surprising Effectiveness of PPO in Cooperative MARL"）
5. **QMIX论文**（Rashid et al., ICML 2018）—— 理解CTDE和单调性约束
6. **COMA论文**（Foerster et al., AAAI 2018）—— 理解counterfactual baseline

**必读代码：**

- CleanRL `ppo_atari.py`（逐行注释，理解每一行为什么这么写）
- CleanRL `ppo_pettingzoo_ma_atari.py`（理解MARL接入方式）
- EPyMARL 的 QMIX 实现（理解mixing network）

**核心概念确认（面试必考，必须能脱稿讲清楚）：**

```
□ GAE(λ)推导：从TD(0)到TD(λ)再到GAE，为什么λ=0.95是常用值
□ PPO clip推导：为什么clip比TRPO的KL约束更好用
□ dual-clip PPO：为什么大批量分布式训练需要再加下界clip
□ Importance Sampling在PPO中的角色：为什么PPO"看起来off-policy但其实是on-policy"
□ CTDE范式：为什么训练时用中心化critic，执行时用分布式actor
□ QMIX的单调性约束：为什么∂Q_tot/∂Q_i ≥ 0，这限制了什么
□ COMA的counterfactual baseline：与普通baseline的区别
□ Self-play为什么能避免reward hacking
□ Curriculum Learning的三种实现方式（环境难度/对手强度/任务分解）
```

#### B.2 算法实现（第3-7周）

**代码架构：**

```
hybrid_arena/
├── algorithms/
│   ├── base.py              # 算法基类（统一接口）
│   ├── networks.py           # 共用网络模块
│   ├── ppo/
│   │   ├── ppo.py           # Vanilla PPO（单智能体，CleanRL风格）
│   │   ├── ppo_dualclip.py  # Dual-clip PPO
│   │   └── config.py
│   ├── mappo/
│   │   ├── mappo.py         # Multi-Agent PPO
│   │   ├── shared_critic.py # 中心化Critic
│   │   └── config.py
│   ├── qmix/
│   │   ├── qmix.py          # QMIX（CTDE, value decomposition）
│   │   ├── mixing_network.py
│   │   └── config.py
│   ├── coma/
│   │   ├── coma.py          # COMA（counterfactual baseline）
│   │   └── config.py
│   └── self_play/
│       ├── manager.py       # Self-play对手池管理
│       ├── elo.py           # ELO评分系统
│       └── curriculum.py    # Curriculum Learning调度器
├── training/
│   ├── trainer.py           # 统一训练循环
│   ├── buffer.py            # Rollout Buffer
│   ├── evaluator.py         # 评估器（对战评估+指标统计）
│   └── logger.py            # W&B日志封装
└── configs/
    ├── ppo_default.yaml
    ├── mappo_default.yaml
    ├── qmix_default.yaml
    └── experiment_sweep.yaml  # 超参搜索配置
```

**网络架构（networks.py）—— 这是面试官最爱问的部分：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MapEncoder(nn.Module):
    """
    处理11×11×11的局部地图观测
    设计思路：
    - 用3层CNN提取空间特征（参考绝悟的channel attention）
    - 最后一层加入spatial attention让网络关注重要区域
    面试话术："为什么用CNN不用MLP？因为地图观测具有空间局部性，
              CNN的参数共享和局部感受野天然适合这种网格结构"
    """
    def __init__(self, in_channels=11, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_dim, 3, padding=1)

        # Spatial Attention（参考绝悟的target attention思想）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Flatten(),
            nn.Softmax(dim=-1),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        # x: (batch, 11, 11, 11) -> (batch, 11, 11, 11) channels first
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        features = F.relu(self.conv3(x))  # (batch, hidden_dim, 11, 11)

        # Spatial attention weighted pooling
        attn_weights = self.spatial_attn(features)  # (batch, 121)
        attn_weights = attn_weights.view(-1, 1, 11, 11)
        attended = (features * attn_weights).sum(dim=[2, 3])  # (batch, hidden_dim)

        return attended


class StateEncoder(nn.Module):
    """编码自身状态(20) + 队友状态(3×15) + 全局信息(10)"""
    def __init__(self, self_dim=20, teammate_dim=15, n_teammates=3,
                 global_dim=10, hidden_dim=64):
        super().__init__()
        self.self_net = nn.Sequential(
            nn.Linear(self_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        # 队友信息用 attention pooling（参考绝悟的multi-head思路）
        self.teammate_net = nn.Sequential(
            nn.Linear(teammate_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        self.teammate_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.global_net = nn.Sequential(
            nn.Linear(global_dim, 32), nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )
        self.output_dim = hidden_dim * 3

    def forward(self, self_state, teammate_states, global_info):
        self_feat = F.relu(self.self_net(self_state))  # (batch, hidden_dim)

        # Teammate attention
        tm_feats = F.relu(self.teammate_net(
            teammate_states.view(-1, teammate_states.shape[-1])
        )).view(teammate_states.shape[0], -1, self.teammate_net[-1].out_features)
        tm_attended, _ = self.teammate_attn(tm_feats, tm_feats, tm_feats)
        tm_feat = tm_attended.mean(dim=1)  # (batch, hidden_dim)

        global_feat = F.relu(self.global_net(global_info))  # (batch, hidden_dim)

        return torch.cat([self_feat, tm_feat, global_feat], dim=-1)


class ActorCritic(nn.Module):
    """
    完整的Actor-Critic网络
    关键设计决策（面试必讲）：
    1. Actor和Critic共享MapEncoder+StateEncoder的前半部分，减少参数量
    2. Critic额外接收全局状态（CTDE的"中心化"体现在这里）
    3. Actor输出三个独立的动作head（移动/技能/目标），而不是一个324维的大softmax
       ——这是因为动作维度之间有弱相关性，独立head训练更稳定
    4. 加入action mask：将不合法动作的logit设为-inf
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.map_encoder = MapEncoder(hidden_dim=hidden_dim)
        self.state_encoder = StateEncoder(hidden_dim=hidden_dim)

        feature_dim = hidden_dim + hidden_dim * 3  # map + state

        # Actor: 三个独立的动作头
        self.move_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 9),  # 8方向 + 不动
        )
        self.skill_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 4),  # 普攻/技能1/技能2/不攻击
        )
        self.target_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 9),  # 8目标 + 不选
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def get_features(self, obs):
        map_feat = self.map_encoder(obs["local_map"])
        state_feat = self.state_encoder(
            obs["self_state"], obs["teammate_states"], obs["global_info"]
        )
        return torch.cat([map_feat, state_feat], dim=-1)

    def get_action_and_value(self, obs, action=None):
        features = self.get_features(obs)

        # Actor logits + action mask
        move_logits = self.move_head(features)
        skill_logits = self.skill_head(features)
        target_logits = self.target_head(features)

        # Apply action mask (关键！不做mask训练会崩)
        action_mask = obs["action_mask"]  # (batch, 324)
        move_mask = action_mask[:, :9]
        skill_mask = action_mask[:, 9:13]
        target_mask = action_mask[:, 13:22]

        move_logits = move_logits.masked_fill(~move_mask.bool(), -1e8)
        skill_logits = skill_logits.masked_fill(~skill_mask.bool(), -1e8)
        target_logits = target_logits.masked_fill(~target_mask.bool(), -1e8)

        move_dist = torch.distributions.Categorical(logits=move_logits)
        skill_dist = torch.distributions.Categorical(logits=skill_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        if action is None:
            move_action = move_dist.sample()
            skill_action = skill_dist.sample()
            target_action = target_dist.sample()
        else:
            move_action = action[:, 0]
            skill_action = action[:, 1]
            target_action = action[:, 2]

        log_prob = (move_dist.log_prob(move_action) +
                    skill_dist.log_prob(skill_action) +
                    target_dist.log_prob(target_action))
        entropy = (move_dist.entropy() +
                   skill_dist.entropy() +
                   target_dist.entropy())

        value = self.critic(features).squeeze(-1)

        action_out = torch.stack([move_action, skill_action, target_action], dim=-1)
        return action_out, log_prob, entropy, value
```

**Dual-clip PPO实现（核心改进点，面试重点讲这个）：**

```python
# algorithms/ppo/ppo_dualclip.py

class DualClipPPO:
    """
    Dual-clip PPO（来自腾讯绝悟论文）

    为什么需要dual-clip：
    标准PPO只clip了ratio的上界（防止步子迈太大），但在大批量分布式训练中，
    由于数据延迟，ratio也可能变得很小（策略变化太快），导致梯度过小训练停滞。
    Dual-clip加了一个下界c（通常=3），当ratio<1/(1+ε)时用c*advantage替换，
    防止"反向过大更新"。

    面试话术：
    "标准clip只防止了positive direction的过大更新，
     dual-clip额外防止了negative direction的过大更新，
     这在分布式多GPU训练中尤其重要，因为actor和learner之间有policy lag"
    """
    def __init__(self, config):
        self.clip_eps = config.get("clip_eps", 0.2)
        self.dual_clip_c = config.get("dual_clip_c", 3.0)  # 绝悟用c=3
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.gamma = config.get("gamma", 0.99)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 256)

        # 自适应熵系数（你的改进点之一）
        self.entropy_schedule = config.get("entropy_schedule", "linear_decay")
        self.entropy_start = config.get("entropy_start", 0.05)
        self.entropy_end = config.get("entropy_end", 0.001)

    def compute_loss(self, batch, network, global_step, total_steps):
        """
        PPO Loss计算，含dual-clip改进

        标准PPO loss:
            L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

        Dual-clip PPO loss (当A < 0时):
            L_dual = max(L_clip, c * A)
            即：当advantage为负时，loss不会低于c*A，防止ratio过小导致的反向过大更新
        """
        obs, actions, old_log_probs, advantages, returns = batch

        # 获取当前策略的log_prob和value
        _, new_log_probs, entropy, new_values = network.get_action_and_value(obs, actions)

        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 标准clip
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        policy_loss = torch.min(surrogate1, surrogate2)

        # === DUAL CLIP（核心改进） ===
        # 当advantage < 0时，加入下界约束
        dual_clip_mask = (advantages < 0).float()
        dual_clip_value = self.dual_clip_c * advantages
        policy_loss = (
            dual_clip_mask * torch.max(policy_loss, dual_clip_value) +
            (1 - dual_clip_mask) * policy_loss
        )
        policy_loss = -policy_loss.mean()

        # Value loss（clip版本，防止value function更新过大）
        value_loss = F.mse_loss(new_values, returns)

        # 自适应熵系数
        entropy_coef = self._get_entropy_coef(global_step, total_steps)
        entropy_loss = -entropy.mean()

        # 总loss
        total_loss = (policy_loss +
                     self.value_loss_coef * value_loss +
                     entropy_coef * entropy_loss)

        return total_loss, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "entropy_coef": entropy_coef,
            "approx_kl": ((ratio - 1) - torch.log(ratio)).mean().item(),
            "clip_fraction": ((ratio - 1).abs() > self.clip_eps).float().mean().item(),
            "dual_clip_fraction": (
                dual_clip_mask * (policy_loss == dual_clip_value).float()
            ).mean().item(),
        }

    def _get_entropy_coef(self, global_step, total_steps):
        """
        自适应熵系数调度（你的改进点之二）

        思路：训练初期用大熵系数鼓励探索（0.05），
              后期用小熵系数让策略收敛（0.001）。
        面试话术："固定熵系数的问题是，初期探索不够导致过早收敛到局部最优，
                  后期探索过多导致策略不稳定。线性衰减是最简单有效的方案，
                  也可以用inverse-sigmoid衰减让中期探索保持更久"
        """
        if self.entropy_schedule == "linear_decay":
            progress = min(global_step / total_steps, 1.0)
            return self.entropy_start + (self.entropy_end - self.entropy_start) * progress
        elif self.entropy_schedule == "cosine_decay":
            progress = min(global_step / total_steps, 1.0)
            return self.entropy_end + 0.5 * (self.entropy_start - self.entropy_end) * \
                   (1 + np.cos(np.pi * progress))
        else:
            return self.entropy_coef
```

**Self-Play + Curriculum Learning（self_play/manager.py）：**

```python
class SelfPlayManager:
    """
    Self-Play对手池管理 + Curriculum Learning

    设计哲学（面试必讲）：
    1. 维护一个历史策略池（最近N个checkpoint），随机从中抽取对手
       ——防止"只会打最新版自己"的过拟合
    2. 新策略必须对历史策略池胜率>55%才入池
       ——保证策略池质量单调递增
    3. Curriculum Learning通过动态调整环境难度实现
       ——不是固定的"easy→medium→hard"，而是根据当前胜率自动调节
    """
    def __init__(self, config):
        self.pool_size = config.get("pool_size", 20)
        self.win_threshold = config.get("win_threshold", 0.55)
        self.n_eval_games = config.get("n_eval_games", 50)

        self.policy_pool = []       # List[PolicyCheckpoint]
        self.elo_ratings = {}       # checkpoint_id -> ELO rating
        self.current_elo = 1000.0

    def get_opponent(self):
        """
        从策略池中采样对手
        采样策略：80%概率选最近5个（保证对最新策略的适应），
                  20%概率均匀采样（保持对历史策略的鲁棒性）
        """
        if not self.policy_pool:
            return None  # 使用rule-based作为初始对手

        if np.random.random() < 0.8 and len(self.policy_pool) >= 5:
            idx = np.random.choice(min(5, len(self.policy_pool)))
            return self.policy_pool[-(idx + 1)]
        else:
            return np.random.choice(self.policy_pool)

    def maybe_add_to_pool(self, current_policy, eval_fn):
        """
        评估当前策略是否有资格入池
        eval_fn: 接收(policy_a, policy_b) 返回 policy_a 的胜率
        """
        if not self.policy_pool:
            self.policy_pool.append(current_policy.clone())
            self.elo_ratings[id(self.policy_pool[-1])] = 1000.0
            return True

        # 对最近3个历史策略评估胜率
        opponents = self.policy_pool[-3:]
        total_wins = 0
        total_games = 0

        for opp in opponents:
            win_rate = eval_fn(current_policy, opp, n_games=self.n_eval_games)
            total_wins += win_rate * self.n_eval_games
            total_games += self.n_eval_games

        avg_win_rate = total_wins / total_games

        if avg_win_rate >= self.win_threshold:
            checkpoint = current_policy.clone()
            self.policy_pool.append(checkpoint)

            # 更新ELO
            for opp in opponents:
                expected = 1 / (1 + 10 ** ((self.elo_ratings.get(id(opp), 1000) -
                                             self.current_elo) / 400))
                self.current_elo += 32 * (avg_win_rate - expected)
            self.elo_ratings[id(checkpoint)] = self.current_elo

            # 保持池大小
            if len(self.policy_pool) > self.pool_size:
                oldest = self.policy_pool.pop(0)
                self.elo_ratings.pop(id(oldest), None)

            return True
        return False


class CurriculumScheduler:
    """
    动态难度调整

    三个维度的难度控制：
    1. 对手强度：rule-based → 弱RL → 中RL → 强RL（self-play）
    2. 环境复杂度：小地图 → 大地图、少技能 → 全技能
    3. 奖励密度：密集奖励（每步都有feedback）→ 稀疏奖励（只有胜负）

    面试话术："Curriculum Learning的关键不是'按计划递增难度'，
              而是'根据当前agent能力自适应调节'。我用agent的胜率
              作为能力指标：胜率>70%就升难度，<30%就降难度"
    """
    def __init__(self):
        self.current_level = 0
        self.levels = [
            # Level 0: 入门
            {"map_size": 16, "team_size": 2, "skills_enabled": False,
             "reward_density": "dense", "opponent": "rule_based"},
            # Level 1: 基础
            {"map_size": 24, "team_size": 3, "skills_enabled": True,
             "reward_density": "dense", "opponent": "rule_based"},
            # Level 2: 标准
            {"map_size": 32, "team_size": 4, "skills_enabled": True,
             "reward_density": "normal", "opponent": "self_play_weak"},
            # Level 3: 困难
            {"map_size": 32, "team_size": 4, "skills_enabled": True,
             "reward_density": "sparse", "opponent": "self_play"},
        ]
        self.win_rate_history = []

    def update(self, win_rate: float):
        self.win_rate_history.append(win_rate)
        # 用最近20局的平均胜率判断是否升/降级
        if len(self.win_rate_history) >= 20:
            recent_wr = np.mean(self.win_rate_history[-20:])
            if recent_wr > 0.70 and self.current_level < len(self.levels) - 1:
                self.current_level += 1
                self.win_rate_history = []
                print(f"[Curriculum] Level up → {self.current_level}")
            elif recent_wr < 0.30 and self.current_level > 0:
                self.current_level -= 1
                self.win_rate_history = []
                print(f"[Curriculum] Level down → {self.current_level}")

    def get_env_config(self):
        return self.levels[self.current_level]
```

#### B.3 统一对比实验（第6-7周）

**实验配置（configs/experiment_sweep.yaml）：**

```yaml
# 所有算法共用的超参数
common:
  total_timesteps: 10_000_000
  n_envs: 8
  gamma: 0.99
  gae_lambda: 0.95
  learning_rate: 3e-4
  n_epochs: 10
  batch_size: 256
  max_grad_norm: 0.5
  seed: [42, 123, 456, 789, 1024]  # 5个seed取平均

# 各算法特定配置
algorithms:
  ppo_vanilla:
    clip_eps: 0.2
    entropy_coef: 0.01
    self_play: false

  ppo_dualclip:
    clip_eps: 0.2
    dual_clip_c: 3.0
    entropy_schedule: linear_decay
    entropy_start: 0.05
    entropy_end: 0.001
    self_play: true
    curriculum: true

  mappo:
    clip_eps: 0.2
    shared_critic: true
    self_play: true

  qmix:
    buffer_size: 50000
    learning_rate: 5e-4
    target_update_freq: 200
    mixing_embed_dim: 32

  coma:
    actor_lr: 5e-4
    critic_lr: 1e-3
    td_lambda: 0.8

# 评估配置
evaluation:
  eval_interval: 50000  # 每5万步评估一次
  n_eval_episodes: 100
  opponents: ["rule_based", "self"]  # 对rule-based和自我对战的胜率
  metrics: ["win_rate", "avg_reward", "avg_episode_length",
            "avg_kills", "avg_deaths", "kda_ratio", "elo"]
```

**训练启动脚本：**

```python
# scripts/train_all.py
"""一键启动所有算法的对比训练"""

import subprocess
import itertools

ALGORITHMS = ["ppo_vanilla", "ppo_dualclip", "mappo", "qmix", "coma"]
SEEDS = [42, 123, 456, 789, 1024]

for algo, seed in itertools.product(ALGORITHMS, SEEDS):
    cmd = [
        "python", "training/train.py",
        "--algorithm", algo,
        "--seed", str(seed),
        "--total_timesteps", "10000000",
        "--wandb_project", "hybrid-arena",
        "--wandb_group", algo,
        "--wandb_name", f"{algo}_seed{seed}",
    ]
    print(f"Launching: {algo} seed={seed}")
    subprocess.Popen(cmd)
```

**消融实验设计（面试官最爱看的表格）：**

```
实验组1：PPO改进消融
┌──────────────────────────────┬─────────┬──────────┬─────────┐
│ 配置                          │ Win Rate │ Avg ELO  │ KDA     │
├──────────────────────────────┼─────────┼──────────┼─────────┤
│ PPO (vanilla, no self-play)  │ 基线     │ 基线      │ 基线    │
│ + dual-clip                  │ +?%      │ +?       │ +?      │
│ + dual-clip + entropy decay  │ +?%      │ +?       │ +?      │
│ + above + self-play          │ +?%      │ +?       │ +?      │
│ + above + curriculum         │ +?%      │ +?       │ +?      │
│ = Full (dual-clip PPO)       │ 最终     │ 最终      │ 最终    │
└──────────────────────────────┴─────────┴──────────┴─────────┘

实验组2：五算法横向对比
┌──────────────────────────────┬─────────┬──────────┬─────────┐
│ 算法                          │ Win Rate │ Avg ELO  │ 训练时间 │
├──────────────────────────────┼─────────┼──────────┼─────────┤
│ Rule-Based                   │ 50.0%   │ 1000     │ N/A     │
│ PPO (vanilla)                │ ?       │ ?        │ ?       │
│ Dual-clip PPO + SP + CL     │ ?       │ ?        │ ?       │
│ MAPPO                        │ ?       │ ?        │ ?       │
│ QMIX                         │ ?       │ ?        │ ?       │
│ COMA                         │ ?       │ ?        │ ?       │
└──────────────────────────────┴─────────┴──────────┴─────────┘
```

#### B.4 阶段B产出检查清单

- [ ] 5种算法全部收敛（至少比rule-based胜率高20%以上）
- [ ] W&B上有完整的训练曲线（5 seeds × 5 algorithms = 25条曲线）
- [ ] 消融表格数据齐全（PPO改进逐步消融 + 五算法横向对比）
- [ ] ELO评分系统正常工作
- [ ] Self-play对手池有效（池中策略ELO单调递增）
- [ ] 发布第一篇知乎博客：「从零手撕dual-clip PPO：为什么腾讯绝悟不用标准PPO」
- [ ] GitHub release tag `v0.2.0-drl`


### ═══════════════════════════════════════════
### 阶段 C：LLM高层规划器
### 预计时间：4-5周
### ═══════════════════════════════════════════

#### C.1 前置学习（第1周）

**必读论文：**
1. ReAct（Yao et al., 2022）—— LLM推理+行动的经典范式
2. Reflexion（Shinn et al., 2023）—— 自我反思改进
3. Voyager（Wang et al., 2023）—— Minecraft中LLM驱动的agent
4. PORTAL（Tencent 2025）—— LLM生成行为树DSL

**必读框架文档：**
- LangGraph官方教程（https://langchain-ai.github.io/langgraph/）
- CrewAI官方文档（https://docs.crewai.com/）

#### C.2 高层规划器架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                  LLM High-Level Planner                     │
│                                                             │
│  输入：                                                      │
│  - 当前游戏状态的自然语言摘要（由 StateTranslator 生成）       │
│  - 最近5个回合的行动历史与结果                                │
│  - 队友的当前意图（多NPC协作信息）                            │
│                                                             │
│  输出：                                                      │
│  - 高层策略指令（如"集合中路开团"/"分兵带线"/"撤退发育"）       │
│  - 每个英雄的角色分配（如"坦克先手开团"/"输出集火后排"）       │
│  - 预期效果与风险评估                                        │
│                                                             │
│  指令被翻译为DRL的reward shaping信号和目标点                  │
│  DRL负责具体的移动、技能释放、目标选择                        │
└─────────────────────────────────────────────────────────────┘
```

**StateTranslator（游戏状态 → 自然语言）：**

```python
class StateTranslator:
    """
    将游戏数值状态翻译为LLM可理解的自然语言

    为什么需要这个中间层：
    1. LLM不擅长处理原始数值（0.73的HP比"血量较低"更难理解）
    2. 自然语言摘要可以过滤掉无关信息，让LLM聚焦决策
    3. 这个翻译层本身就是可调优的（用GRPO训练时，翻译质量也会提升）
    """
    def translate(self, game_state: GameState, team: str) -> str:
        heroes = self._get_team_heroes(game_state, team)
        enemies = self._get_visible_enemies(game_state, team)

        prompt_parts = []

        # 当前局势
        prompt_parts.append(f"## 当前局势 (第{game_state.step_count}步)")
        prompt_parts.append(f"我方击杀:{game_state.red_kills if team=='red' else game_state.blue_kills} "
                          f"敌方击杀:{game_state.blue_kills if team=='red' else game_state.red_kills}")

        # 我方状态
        prompt_parts.append("\n## 我方队伍状态")
        for h in heroes:
            hp_desc = self._hp_description(h.hp / h.max_hp)
            mp_desc = self._mp_description(h.mp / h.max_mp)
            cd_desc = self._cd_description(h)
            prompt_parts.append(
                f"- {h.role}({h.agent_id}): {hp_desc}, {mp_desc}, "
                f"位于({h.x},{h.y}), {cd_desc}"
            )

        # 可见敌方
        prompt_parts.append(f"\n## 可见敌方 ({len(enemies)}人)")
        for e in enemies:
            prompt_parts.append(
                f"- 敌方{e.role}: {self._hp_description(e.hp/e.max_hp)}, 位于({e.x},{e.y})"
            )
        if len(enemies) < 4:
            prompt_parts.append(f"（有{4-len(enemies)}名敌人在视野外，注意防守）")

        # 建筑状态
        prompt_parts.append(f"\n## 建筑: 我方塔{self._tower_status(game_state, team)}, "
                          f"敌方塔{self._tower_status(game_state, 'blue' if team=='red' else 'red')}")

        return "\n".join(prompt_parts)

    def _hp_description(self, ratio):
        if ratio > 0.8: return "血量充足"
        elif ratio > 0.5: return "血量一般"
        elif ratio > 0.2: return "血量较低⚠️"
        else: return "危险！血量极低🚨"
```

**LangGraph 状态机实现：**

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Literal
import json

# === 状态定义 ===
class TeamState(BaseModel):
    game_summary: str = ""
    action_history: List[str] = Field(default_factory=list)
    current_strategy: str = "balanced"
    hero_assignments: dict = Field(default_factory=dict)
    reflection: str = ""
    turn_count: int = 0

# === 节点函数 ===
def analyze_situation(state: TeamState) -> TeamState:
    """节点1：分析当前局势"""
    prompt = f"""你是一个MOBA游戏的战术分析师。
分析以下游戏状态，判断当前局势是"优势"、"劣势"还是"均势"。

{state.game_summary}

最近行动历史：
{chr(10).join(state.action_history[-5:])}

请简洁分析（50字以内），输出JSON格式：
{{"situation": "优势/劣势/均势", "reason": "分析原因"}}"""

    response = llm.invoke(prompt)
    # parse response...
    return state

def decide_strategy(state: TeamState) -> TeamState:
    """节点2：制定高层策略（ReAct风格）"""
    prompt = f"""你是一个MOBA游戏的战术指挥官。

当前局势分析：{state.reflection}
当前游戏状态：
{state.game_summary}

可选策略：
1. "团战" - 集合队伍寻找有利位置发起团战
2. "分推" - 分散兵力同时推进多路
3. "发育" - 优先清野和打小兵获取经济
4. "防守" - 收缩防线保护建筑
5. "抓人" - 利用视野优势找落单敌人击杀

请根据局势选择最优策略，并为每个英雄分配具体任务。

输出JSON格式：
{{
  "strategy": "团战/分推/发育/防守/抓人",
  "reasoning": "选择理由（30字以内）",
  "assignments": {{
    "tank": "具体任务描述",
    "dps_1": "具体任务描述",
    "dps_2": "具体任务描述",
    "support": "具体任务描述"
  }},
  "target_positions": {{
    "tank": [x, y],
    "dps_1": [x, y],
    "dps_2": [x, y],
    "support": [x, y]
  }}
}}"""

    response = llm.invoke(prompt)
    parsed = json.loads(response.content)
    state.current_strategy = parsed["strategy"]
    state.hero_assignments = parsed["assignments"]
    return state

def reflect_on_outcome(state: TeamState) -> TeamState:
    """节点3：Reflexion — 回顾上一轮策略的效果"""
    prompt = f"""你是一个MOBA游戏的战术复盘师。

上一轮策略：{state.current_strategy}
上一轮各英雄任务：{json.dumps(state.hero_assignments, ensure_ascii=False)}
执行结果：{state.action_history[-1] if state.action_history else '无'}

请分析：
1. 上一轮策略是否有效？为什么？
2. 有什么可以改进的？
3. 是否需要改变策略方向？

简洁回答（100字以内）。"""

    response = llm.invoke(prompt)
    state.reflection = response.content
    return state

def should_reflect(state: TeamState) -> str:
    """条件边：是否需要反思（每5个回合反思一次）"""
    if state.turn_count > 0 and state.turn_count % 5 == 0:
        return "reflect"
    return "decide"

# === 构建状态机 ===
workflow = StateGraph(TeamState)
workflow.add_node("analyze", analyze_situation)
workflow.add_node("reflect", reflect_on_outcome)
workflow.add_node("decide", decide_strategy)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", should_reflect, {
    "reflect": "reflect",
    "decide": "decide",
})
workflow.add_edge("reflect", "decide")
workflow.add_edge("decide", END)

planner = workflow.compile()
```

**策略指令 → DRL目标转换器：**

```python
class StrategyToRLBridge:
    """
    将LLM的高层策略指令转换为DRL可用的信号

    转换方式（两种互补）：
    1. 目标点注入：将LLM指定的目标位置作为额外奖励信号
       r_goal = -α × distance_to_target_point
    2. Reward Shaping：根据策略类型调整奖励权重
       "团战"策略下，r_kill权重×2、r_farm权重×0.5
       "发育"策略下，r_farm权重×2、r_kill权重×0.5
    """
    def translate_to_reward_modifier(self, strategy: str, assignments: dict) -> dict:
        modifiers = {
            "团战": {
                "kill": 2.0, "death": 1.5, "assist": 2.0,
                "tower": 1.0, "farm": 0.3, "grouping_bonus": 0.5
            },
            "分推": {
                "kill": 0.5, "death": 0.5, "assist": 0.3,
                "tower": 3.0, "farm": 1.5, "spread_bonus": 0.3
            },
            "发育": {
                "kill": 0.3, "death": 2.0, "assist": 0.2,
                "tower": 0.3, "farm": 3.0, "safe_farming_bonus": 0.5
            },
            "防守": {
                "kill": 0.5, "death": 2.5, "assist": 1.0,
                "tower": 0.5, "tower_lost": 3.0, "near_tower_bonus": 0.5
            },
            "抓人": {
                "kill": 3.0, "death": 1.0, "assist": 1.5,
                "tower": 0.5, "farm": 0.5, "isolation_bonus": 0.5
            },
        }
        return modifiers.get(strategy, modifiers["团战"])

    def get_goal_positions(self, assignments: dict, hero_configs: dict) -> dict:
        """将LLM分配的目标位置转化为DRL的导航目标"""
        goals = {}
        for role, (x, y) in assignments.get("target_positions", {}).items():
            goals[role] = np.array([x, y], dtype=np.float32)
        return goals
```

#### C.3 多NPC角色化（CrewAI集成，可选增强）

```python
from crewai import Agent, Task, Crew

# 为4个英雄创建不同"性格"的AI角色
tank_agent = Agent(
    role="前线指挥官",
    goal="保护队友并创造开团机会",
    backstory="你是一个经验丰富的坦克玩家，擅长判断团战时机。"
              "你的核心职责是在团战中先手控制敌方核心输出，"
              "并在队友血量低时用技能掩护撤退。",
    verbose=False,
    llm=qwen_model,
)

dps_agent = Agent(
    role="输出核心",
    goal="最大化伤害输出并存活",
    backstory="你是一个冷静的输出位，只在确认安全时才输出。"
              "你的核心原则是：1) 永远不打第一个进场，"
              "2) 优先攻击敌方后排，3) 保留闪避技能用于自保。",
    verbose=False,
    llm=qwen_model,
)
```

#### C.4 阶段C产出检查清单

- [ ] LangGraph状态机正常运行（分析→反思→决策流程完整）
- [ ] StateTranslator能正确翻译游戏状态为自然语言
- [ ] LLM策略指令能正确转换为DRL奖励信号
- [ ] Hybrid模式（LLM+DRL）对纯DRL模式的胜率对比完成
- [ ] 发布第二篇知乎博客：「让AI学会"开团时机"：用LLM做MOBA游戏的战术大脑」
- [ ] GitHub release tag `v0.3.0-hybrid`


### ═══════════════════════════════════════════
### 阶段 D：Agentic RL —— GRPO微调LLM Planner
### 预计时间：4-5周
### ═══════════════════════════════════════════

#### D.1 核心思想

```
传统方式：LLM planner 用 prompt engineering 固定不变
              ↓
我们的改进：用 GRPO（Group Relative Policy Optimization）
           对 LLM planner 做多回合强化学习微调
              ↓
训练信号来源：每局游戏的胜负结果 + 策略执行后的即时效果
```

**为什么用GRPO而不是PPO来训练LLM：**

```
PPO训练LLM需要一个单独的critic/value model（参数量=policy模型），
计算量翻倍。GRPO通过组内相对比较计算优势函数，省去了critic：

  GRPO advantage:
    A_i = (R_i - mean(R_group)) / std(R_group)

  其中 R_group 是同一个prompt下采样多个response的奖励集合。
  这比PPO的V(s)估计更稳定，且不需要额外的value网络。

面试话术：
"GRPO的核心洞察是：在LLM场景下，同一个状态可以并行采样多个策略输出，
 用组内对比替代value function估计。这在DeepSeek-R1中被验证是可行的。
 我将这个思路迁移到游戏策略LLM的训练中。"
```

#### D.2 GRPO训练管线实现

```python
# training/grpo_trainer.py
import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

class GameGRPOTrainer:
    """
    用GRPO训练LLM战术规划器

    训练流程：
    1. LLM接收游戏状态 → 输出策略指令
    2. 策略指令交给DRL执行 → 得到游戏结果（胜/负/KDA等）
    3. 游戏结果作为reward → GRPO更新LLM参数
    4. 重复1-3

    每个训练step：
    - 同一个游戏状态，让LLM生成G=8组不同策略
    - 每组策略各打一局（或模拟N步）
    - 用胜率/KDA作为reward
    - 计算组内相对advantage → 更新LLM
    """
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        self.grpo_config = GRPOConfig(
            output_dir="./grpo_output",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-6,
            num_generations=8,         # 每个prompt采样8个response
            max_new_tokens=256,
            temperature=0.7,
            beta=0.1,                  # KL正则化系数
            logging_steps=1,
            report_to="wandb",
        )

    def compute_reward(self, strategy_output: str, game_state: dict,
                       drl_executor, n_simulations: int = 3) -> float:
        """
        执行LLM策略并计算奖励

        奖励 = 0.5 × win_rate + 0.3 × kda_score + 0.2 × objective_score
        """
        try:
            parsed = json.loads(strategy_output)
            strategy = parsed.get("strategy", "团战")
            assignments = parsed.get("assignments", {})
        except json.JSONDecodeError:
            return -1.0  # 格式错误惩罚

        total_reward = 0.0
        for _ in range(n_simulations):
            # 用DRL在当前策略指导下打一局
            result = drl_executor.simulate_game(
                strategy=strategy,
                assignments=assignments,
                game_state=game_state,
                max_steps=200,  # 模拟200步而非完整一局（加速训练）
            )
            # 计算多维度奖励
            win_bonus = 1.0 if result["win"] else 0.0
            kda = (result["kills"] + result["assists"]) / max(result["deaths"], 1)
            kda_score = min(kda / 5.0, 1.0)  # 归一化到[0,1]
            obj_score = result["towers_taken"] * 0.3 + result["farm_efficiency"] * 0.2

            reward = 0.5 * win_bonus + 0.3 * kda_score + 0.2 * obj_score
            total_reward += reward

        return total_reward / n_simulations

    def build_training_dataset(self, game_states: list) -> list:
        """
        构建GRPO训练数据

        每条数据 = {prompt: 游戏状态描述, completion: 策略JSON, reward: 游戏结果}
        """
        dataset = []
        translator = StateTranslator()

        for state in game_states:
            prompt = self._build_prompt(translator.translate(state, "red"), state)
            dataset.append({
                "prompt": prompt,
                "state": state,  # 用于后续计算reward
            })

        return dataset

    def _build_prompt(self, state_description: str, state: dict) -> str:
        return f"""你是一个专业的MOBA游戏AI战术指挥官。

{state_description}

请分析局势并制定最优战术方案。输出JSON格式：
{{
  "strategy": "团战/分推/发育/防守/抓人",
  "reasoning": "选择理由",
  "assignments": {{
    "tank": "任务描述",
    "dps_1": "任务描述",
    "dps_2": "任务描述",
    "support": "任务描述"
  }},
  "target_positions": {{
    "tank": [x, y],
    "dps_1": [x, y],
    "dps_2": [x, y],
    "support": [x, y]
  }}
}}

只输出JSON，不要其他内容。"""
```

#### D.3 最终对比实验矩阵

```
┌───────────────────────────────────────┬─────────┬──────────┬──────────────┐
│ 方案                                   │ Win Rate │ Avg ELO  │ 策略合理性评分 │
├───────────────────────────────────────┼─────────┼──────────┼──────────────┤
│ A. Pure Rule-Based (FSM)             │ 50.0%   │ 1000     │ N/A          │
│ B. Pure DRL (best single algo)       │ ?       │ ?        │ N/A          │
│ C. LLM(prompt-only) + DRL           │ ?       │ ?        │ 人工评估1-5分  │
│ D. LLM(ReAct) + DRL                 │ ?       │ ?        │ 人工评估1-5分  │
│ E. LLM(Reflexion) + DRL             │ ?       │ ?        │ 人工评估1-5分  │
│ F. LLM(GRPO-trained) + DRL ★        │ ?       │ ?        │ 人工评估1-5分  │
└───────────────────────────────────────┴─────────┴──────────┴──────────────┘

关键发现（预期）：
- F > E > D > C > B >> A
- GRPO训练后的LLM在"团战时机判断"上准确率显著提升
- 纯DRL在"微操"上强于LLM指导版本，但"大局观"差距明显
```

#### D.4 阶段D产出检查清单

- [ ] GRPO训练管线跑通（Qwen2.5-1.5B用于快速验证，7B用于最终结果）
- [ ] GRPO训练后的LLM planner胜率明显优于prompt-only版本
- [ ] 六组方案的完整对比表格（含±标准差）
- [ ] 策略可视化：用Streamlit展示LLM的chain-of-thought推理过程
- [ ] 发布第三篇知乎博客：「用GRPO训练LLM学会MOBA战术：从DeepSeek-R1到游戏AI」
- [ ] GitHub release tag `v1.0.0`


### ═══════════════════════════════════════════
### 阶段 E：打磨、部署与简历包装
### 预计时间：2-3周
### ═══════════════════════════════════════════

#### E.1 在线Demo搭建

```python
# demo/app.py (Streamlit)
import streamlit as st

st.title("🎮 HybridArena: LLM × DRL 混合智能体对战平台")

st.sidebar.header("配置")
algo = st.sidebar.selectbox("DRL算法", ["PPO", "Dual-clip PPO", "MAPPO", "QMIX"])
llm_mode = st.sidebar.selectbox("LLM模式", ["关闭", "Prompt-only", "ReAct", "GRPO-trained"])
opponent = st.sidebar.selectbox("对手", ["Rule-Based", "Self-Play Best"])

if st.button("开始对战"):
    # 实时渲染对战过程
    game_container = st.empty()
    thought_container = st.expander("🧠 LLM战术思考过程", expanded=True)

    for step_data in run_game(algo, llm_mode, opponent):
        # 渲染游戏画面
        game_container.image(step_data["frame"], caption=f"Step {step_data['step']}")

        # 展示LLM的思考过程（每5步更新一次）
        if step_data.get("llm_thought"):
            with thought_container:
                st.markdown(f"**策略**: {step_data['llm_thought']['strategy']}")
                st.markdown(f"**理由**: {step_data['llm_thought']['reasoning']}")
                for role, task in step_data['llm_thought']['assignments'].items():
                    st.markdown(f"- {role}: {task}")

    # 显示最终结果
    st.success(f"对战结束！{'胜利 🎉' if step_data['win'] else '失败'}")
    st.metric("最终ELO", step_data["elo"])
    st.metric("KDA", f"{step_data['kills']}/{step_data['deaths']}/{step_data['assists']}")
```

#### E.2 GitHub README模板

```markdown
# 🏟️ HybridArena

**LLM-Decision × DRL-Control: A Hybrid Agent Training Platform
for Imperfect-Information Strategy Games**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![W&B Report](https://img.shields.io/badge/W%26B-Report-orange)]()
[![Demo](https://img.shields.io/badge/🤗-Live%20Demo-green)]()

> 在自研的4v4不完全信息MOBA环境中，用LLM做高层战略决策，
> 用DRL做低层微操控制，并用GRPO对LLM planner进行多回合RL微调。

## ✨ 亮点

- 🎮 **自研MiniMOBA-4v4环境**：PettingZoo API，不完全信息，3种英雄，6k+ steps/sec
- ⚔️ **5种DRL算法统一对比**：PPO / Dual-clip PPO / MAPPO / QMIX / COMA
- 🧠 **LLM高层规划器**：LangGraph状态机 + ReAct + Reflexion
- 🔥 **GRPO训练LLM Planner**：首个将DeepSeek-R1的GRPO用于游戏战术LLM训练的开源项目
- 📊 **严谨消融实验**：25组实验（5算法×5seeds），完整W&B报告

## 📈 主要结果

| 方案 | Win Rate vs Rule-Based | ELO |
|------|----------------------|-----|
| Rule-Based | 50.0% | 1000 |
| PPO (vanilla) | XX.X% | XXXX |
| Dual-clip PPO + Self-Play | XX.X% | XXXX |
| MAPPO | XX.X% | XXXX |
| Hybrid: LLM(GRPO) + DRL | **XX.X%** | **XXXX** |

## 🚀 快速开始

(安装与运行命令)

## 🏗️ 架构

(架构图)

## 📝 技术博客

1. [从零手撕dual-clip PPO](link)
2. [用LLM做MOBA的战术大脑](link)
3. [GRPO训练LLM学会游戏战术](link)

## 📚 参考文献

(论文列表)
```

#### E.3 简历项目描述模板

```
项目名称：HybridArena — LLM×DRL 混合智能体训练平台

项目描述：
设计并开源了面向不完全信息策略游戏的"LLM高层规划+DRL微操控制"混合智能体平台。
在自研的4v4 PettingZoo MOBA环境（6k+ env steps/sec）上：
(1) 实现并对比PPO/dual-clip PPO/MAPPO/QMIX/COMA五种算法，
    dual-clip PPO配合self-play与curriculum learning达到ELO XXXX（+XXX vs vanilla PPO）；
(2) 用LangGraph构建ReAct/Reflexion风格的LLM战术规划器，
    解决纯DRL在"开团时机/分推决策"等宏观策略上的不足；
(3) 复现GRPO对Qwen2.5-7B进行多回合RL微调，
    训练后的LLM planner对纯prompt版本胜率提升X%，
    Hybrid方案整体对纯DRL baseline ELO +XXX。
25组对照实验（5算法×5seeds）全部公开于W&B。

技术栈：PyTorch / PettingZoo / CleanRL / LangGraph / TRL(GRPO) /
        vLLM / Qwen2.5 / W&B / Docker

GitHub: xxx stars | 在线Demo: xxx
```

---

## 三、完整学习路线时间表

```
Month 1 (Week 1-4):
├── Week 1-2: PettingZoo/Gymnasium深入 + 环境核心开发
├── Week 3:   环境测试 + Rule-Based baseline
├── Week 4:   PPO原理精读 + CleanRL代码逐行研读
│
Month 2 (Week 5-8):
├── Week 5:   实现Vanilla PPO + Dual-clip PPO（在自研环境上）
├── Week 6:   实现MAPPO + QMIX
├── Week 7:   实现Self-Play + Curriculum Learning
├── Week 8:   跑完所有对比实验 + 写第一篇博客
│
Month 3 (Week 9-12):
├── Week 9:   LangGraph/CrewAI 学习 + ReAct/Reflexion论文精读
├── Week 10:  实现StateTranslator + LangGraph状态机
├── Week 11:  Hybrid模式联调 + 策略→DRL桥接层
├── Week 12:  LLM vs DRL对比实验 + 写第二篇博客
│
Month 4 (Week 13-16):
├── Week 13:  GRPO/RLHF原理学习 + DeepSeek-R1论文精读
├── Week 14:  GRPO训练管线搭建（先用1.5B模型验证）
├── Week 15:  7B模型GRPO训练 + 最终实验
├── Week 16:  Demo搭建 + README打磨 + 写第三篇博客
│
Month 5 (Week 17-18, 可选):
├── Week 17:  Docker化 + CI/CD + 代码清理
├── Week 18:  简历包装 + 模拟面试准备
```

---

## 四、面试准备：20个高频问题与你的回答框架

### 4.1 关于PPO/DRL

**Q1: PPO的clip机制是如何工作的？为什么比TRPO好？**
> 你的回答框架：从TRPO的KL约束 → 近似为clip → 解释ε的含义 → 引出你做了dual-clip改进

**Q2: 什么是dual-clip PPO？为什么绝悟要用它？**
> 你的回答框架：标准clip只限上界 → 大批量分布式训练中ratio可能过小 → 加下界c防止反向过大更新 → 展示你的消融实验结果

**Q3: GAE(λ)的λ怎么选？**
> 你的回答框架：λ=0是TD(0)（低方差高偏差）→ λ=1是MC（高方差低偏差）→ λ=0.95是经验值 → 展示你在不同λ下的实验结果

**Q4: Self-play如何防止reward hacking？**
> 你的回答框架：固定对手 → agent可能学到"漏洞" → self-play让漏洞被自己利用 → 策略池保证鲁棒性 → 展示你的ELO曲线

### 4.2 关于MARL

**Q5: CTDE范式是什么？为什么需要中心化Critic？**
> 训练时用全局信息（中心化）→ 执行时只用局部观测（分布式）→ 解决非平稳性

**Q6: QMIX的单调性约束限制了什么？**
> ∂Q_tot/∂Q_i ≥ 0 → 无法表达"某些agent的Q值降低反而让整体更好"的情况 → 这是QPLEX/WQMIX改进的出发点

**Q7: COMA和QMIX的credit assignment方式有什么区别？**
> QMIX：通过value decomposition，让每个agent有独立的Q值 → COMA：用counterfactual baseline，计算"如果我做了不同动作，整体奖励会变多少"

### 4.3 关于LLM Agent

**Q8: 为什么不直接用LLM做所有决策，还要加DRL？**
> LLM擅长高层推理和规划但响应太慢 → DRL擅长快速反应但缺乏大局观 → Hybrid取长补短 → 展示你的对比实验

**Q9: GRPO和PPO训练LLM有什么区别？**
> PPO需要value model → GRPO用组内对比 → 省一半计算量 → 更稳定 → DeepSeek-R1验证了可行性

**Q10: LangGraph相比LangChain有什么优势？**
> LangChain是链式 → LangGraph是图状态机 → 支持条件分支和循环 → 天然适合游戏的"分析→决策→执行→反思"循环

### 4.4 关于项目本身

**Q11: 为什么选择自研环境而不用现成的StarCraft/Dota？**
**Q12: 你的环境和OpenAI Five/绝悟有什么本质区别？**
**Q13: 如果要把这个系统部署到真正的游戏中，需要解决什么问题？**
**Q14: 你做的消融实验中，最让你意外的发现是什么？**
**Q15: 如果给你更多算力，你会做什么改进？**

（以上问题的回答框架建议在博客中预先展开，面试时可以自信地引用。）

---

## 五、副项目简要方案（可选，进一步提升简历深度）

### 副项目1：Deep CFR + ESCHER（博弈论深度）

```
目标：在Leduc Hold'em上实现Deep CFR + ESCHER方差减少
时间：2-3周
产出：exploitability降到比OpenSpiel内置实现更低
简历写法："自实现Deep CFR并集成ESCHER零方差估计器，
          在Leduc Hold'em上exploitability达到X mbb/g，
          优于OpenSpiel内置实现Y%"
```

### 副项目2：LightZero上的EfficientZero V2（Model-based深度）

```
目标：在LightZero框架上复现EfficientZero的两个核心改进
时间：2-3周
产出：Atari-100k子集上样本效率对比
简历写法："基于LightZero复现EfficientZero的自监督一致性损失与
          value-prefix LSTM，在Atari-100k子集上样本效率
          较vanilla MuZero提升X倍"
```

---

## 附录A：关键论文阅读清单（按优先级排序）

```
★★★ 必读（做项目前必须读完）
1. Schulman et al., 2017 — PPO
2. Ye et al., 2020 (AAAI) — 王者绝悟1v1 (dual-clip PPO)
3. Yu et al., 2022 — MAPPO ("The Surprising Effectiveness of PPO")
4. Rashid et al., 2018 (ICML) — QMIX
5. Yao et al., 2022 — ReAct
6. Shao et al., 2024 — DeepSeek-R1 (GRPO)

★★ 强烈推荐（做到阶段C/D时读）
7. Foerster et al., 2018 (AAAI) — COMA
8. Shinn et al., 2023 — Reflexion
9. Wang et al., 2023 — Voyager
10. Luo et al., 2025 — SPIRAL
11. Ye et al., 2021 — EfficientZero

★ 推荐（时间充裕时读）
12. Espeholt et al., 2018 — IMPALA (V-trace)
13. Schrittwieser et al., 2020 — MuZero
14. Brown et al., 2019 — Deep CFR
15. Farina et al., 2023 — ESCHER
```

## 附录B：硬件需求估算

```
最低配置（全部可跑，但训练时间较长）：
- GPU: 1× RTX 3090 (24GB)
- CPU: 8核+
- RAM: 32GB
- 存储: 100GB SSD
- 训练时间估算: 阶段B约3-4天，阶段D（7B GRPO）约2-3天

推荐配置（舒适开发）：
- GPU: 2× RTX 4090 (48GB total)
- CPU: 16核+
- RAM: 64GB
- 训练时间估算: 阶段B约1天，阶段D约1天

云服务替代方案：
- AutoDL / 矩池云：RTX 4090 约2-3元/小时
- 阶段B约100-150元，阶段D约150-200元
- 总预算约300-500元可完成所有训练
```

---

> **最后一句话**：这个项目的核心竞争力不在于"做了很多"，而在于**每一步都有明确的WHY**。面试官真正想听的不是"我用了PPO"，而是"我用了dual-clip PPO，因为在分布式训练中标准clip会导致X问题，我的消融实验证明了Y结果"。把这个思维贯穿到项目的每一个设计决策中，你就能在面试中讲出别人讲不出的深度。
