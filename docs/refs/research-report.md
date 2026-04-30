# HybridArena 下一阶段技术调研报告

生成日期：2026-04-29

## 1. 调研范围

本报告服务于 HybridArena 的下一步完善，不重新规划整个旗舰项目，而是聚焦以下可落地技术点：

- PettingZoo Parallel API 合规性
- MultiDiscrete / joint action mask 的实现策略
- PPO / DualClipPPO 工程细节修复
- 多环境采样与训练吞吐
- LLM Planner MVP 与后续 GRPO 训练入口

## 2. 技术方案对比

| 方案 | 来源 | 核心原理 | 优势 | 劣势 | 适用场景 | 可行性评分 |
|---|---|---|---|---|---|---|
| 324-way joint categorical policy | 当前动作空间 `9×4×9=324` | 三个 head 产生 move/skill/target logits 后广播相加成 `(B,324)`，再用 full `action_mask` 采样 | 精确匹配现有 `(324,)` mask；改动集中；训练/采样一致 | 不能显式建模 skill→target 条件结构 | 当前 HybridArena 的 P0 修复 | 5 |
| Conditional factorized policy | 自定义策略实现 | 先采样 move/skill，再根据已选 skill 构造 target mask | 更贴近 MultiDiscrete 结构；可解释性强 | PPO 训练时 log_prob 复算复杂；测试成本更高 | 后续优化 | 3 |
| 直接 324 维 actor head | Actor 输出一个 324 维 logits | 简单直接 | 放弃 move/skill/target 分头结构；策略可解释性下降 | 快速原型 | 4 |
| 单环境 8 agent 采样 | 当前实现 | 一个 ParallelEnv，每步收集 8 个 agent transition | 实现简单 | 采样吞吐低；`num_envs` 配置失效 | 调试阶段 | 3 |
| 多 ParallelEnv 同步向量化 | 自实现 `ParallelEnvVector` | 创建 N 个 MiniMOBAEnv，同步 step，展平为 `N×agents` batch | 无新依赖；贴合项目结构；可控制 seed | 需要处理 reset 和 agent key 映射 | 下一阶段推荐 | 5 |
| Gymnasium VectorEnv 包装 | Gymnasium SyncVectorEnv / AsyncVectorEnv | 批量环境 API | 标准化，适合单智能体 Gym 环境 | 对 PettingZoo multi-agent dict 需要额外适配 | SingleAgentWrapper 训练 | 3 |
| LangGraph planner | LangGraph StateGraph | 用状态图组织高层策略：observe→plan→tool→reflect | 状态清晰；适合长流程 Agent | 第一版不应引入复杂多 Agent 框架 | LLM Planner MVP | 5 |
| CrewAI 多角色 NPC | Crew / Agent / Task | 为英雄角色配置不同 Agent | 展示效果强 | 对胜率提升不一定明显；依赖多 | Demo 增强，不是 P0/P1 | 2 |
| TRL GRPOTrainer | Hugging Face TRL | 用 GRPO 对 LLM planner 做 post-training | 与 README 阶段 D 对齐 | 需要数据集、reward function、显存策略 | P3/P4 后续 | 3 |

## 3. 推荐方案

### 首选路线

1. 用 **324-way joint categorical policy** 修复 action mask 和 PPO log_prob 一致性。
2. 在 buffer 中保存 `action_mask` 和 `old_values`，修复 PPO clipped value loss。
3. 增加 tower/base/economy objective，让环境从团战模拟升级为 objective-based MiniMOBA。
4. 增加 evaluator、checkpoint、CLI、ablation runner，使阶段 B 可复现。
5. 在 RL baseline 稳定后，实现 LangGraph 风格的 LLM Planner MVP，仅做高层 macro action，不直接控制底层微操。

### 暂缓路线

- 暂缓 CrewAI 多角色化。
- 暂缓 GRPO/QLoRA。
- 暂缓 MAPPO/QMIX/COMA 全量实现。

原因：当前最高风险不是模型能力，而是环境目标与 PPO 训练闭环尚未完全可信。

## 4. 关键参考资料

### PettingZoo Parallel API

- 来源：https://pettingzoo.farama.org/api/parallel/
- 要点：ParallelEnv 适合所有当前 agent 同时 step 的环境；`reset` 返回 observation/info 字典；`step(actions)` 返回 observations、rewards、terminations、truncations、infos。
- 对应模块：`hybrid_arena/minimoba/env.py`

### PettingZoo Action Masking

- 来源：https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/
- 要点：action masking 用于避免 invalid actions，比让非法动作 no-op 更自然。
- 对应模块：`hybrid_arena/minimoba/game_engine.py`、`hybrid_arena/algorithms/networks.py`

### Gymnasium VectorEnv

- 来源：https://gymnasium.farama.org/api/vector/
- 要点：SyncVectorEnv / AsyncVectorEnv 通过批量 reset/step 提升采样吞吐；数据按 `num_envs` 维度批量返回。
- 对应模块：可参考但不直接套用；HybridArena 更适合自实现 multi-agent vector runner。

### CleanRL PPO

- 来源：https://docs.cleanrl.dev/rl-algorithms/ppo/
- 要点：PPO 需要严格处理 rollout log_prob、old values、advantage normalization、clip fraction、KL、entropy 等训练指标。
- 对应模块：`hybrid_arena/algorithms/ppo/`、`hybrid_arena/training/`

### LangGraph

- 来源：https://docs.langchain.com/oss/python/langgraph/overview
- 要点：LangGraph 是面向长时、有状态 Agent 的底层 orchestration framework。
- 对应模块：`hybrid_arena/inference/`

### Hugging Face TRL GRPOTrainer

- 来源：https://huggingface.co/docs/trl/grpo_trainer
- 要点：TRL 支持 GRPOTrainer，可用于语言模型的 GRPO 训练。
- 对应模块：后续 `hybrid_arena/training/grpo_qlora_trainer.py`

## 5. 蒸馏给 Codex 的技术要点

### 5.1 Joint action policy

实现原则：保留三个 actor head，但不要把 `(324,)` mask 错切成三个 mask。

伪代码：

```python
move_logits = move_head(features)      # (B, 9)
skill_logits = skill_head(features)    # (B, 4)
target_logits = target_head(features)  # (B, 9)

joint_logits = (
    move_logits[:, :, None, None]
    + skill_logits[:, None, :, None]
    + target_logits[:, None, None, :]
).reshape(batch_size, 324)

joint_logits = joint_logits.masked_fill(action_mask <= 0, -1e8)
dist = Categorical(logits=joint_logits)
flat_action = dist.sample()
move = flat_action // 36
skill = (flat_action % 36) // 9
target = flat_action % 9
action = torch.stack([move, skill, target], dim=-1)
log_prob = dist.log_prob(flat_action)
entropy = dist.entropy()
```

训练复算 log_prob 时：

```python
flat_action = actions[:, 0] * 36 + actions[:, 1] * 9 + actions[:, 2]
log_prob = dist.log_prob(flat_action)
```

### 5.2 Buffer 必须保存 mask 与 old values

`RolloutBuffer.add()` 增加：

```python
action_mask_batch: np.ndarray  # (N_agents, 324)
```

`get_batch()` 返回：

```python
"action_masks": Tensor[(T*N_agents), 324]
"old_values": Tensor[(T*N_agents)]
```

### 5.3 PPO clipped value loss

```python
value_pred_clipped = old_values + torch.clamp(
    new_values - old_values,
    -clip_eps,
    clip_eps,
)
value_losses = (new_values - returns) ** 2
value_losses_clipped = (value_pred_clipped - returns) ** 2
value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
```

### 5.4 MiniMOBA objective

需要增加 tower/base runtime state：

```python
@dataclass
class StructureState:
    structure_id: str
    team: str
    structure_type: Literal["tower", "base"]
    x: int
    y: int
    hp: float
    max_hp: float
    attack_range: int
    attack_damage: float
    alive: bool
```

击杀与推塔都要更新队伍经济：

```python
if killer.team == "red":
    self.red_gold += gold_reward
else:
    self.blue_gold += gold_reward
```

### 5.5 LLM Planner MVP

LLM 不直接输出底层 `[move, skill, target]`。第一版只输出 macro action：

```python
MacroAction = Literal[
    "group_mid", "push_nearest_tower", "retreat", "farm_safe",
    "protect_support", "force_teamfight", "split_push"
]
```

然后由 `MacroActionAdapter` 转成 rule-based bias 或底层 RL policy 条件输入。
