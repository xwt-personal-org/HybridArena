# HybridArena 下一步完善开发计划

生成日期：2026-04-29

## 元信息

- 项目：HybridArena
- 当前状态：MiniMOBA 环境、Rule-based baseline、PPO/DualClipPPO 训练雏形已实现；LLM Planner 与 GRPO 尚未实现。
- 技术栈：Python 3.10+、PettingZoo、Gymnasium、NumPy、PyTorch、Pygame、Pytest、Ruff；后续 LLM 可选 Transformers / LangGraph / TRL。
- 路线：复杂路线。
- 总模块数：7
- 预计步骤总数：41
- 建议开发顺序：模块 0 → 模块 1 → 模块 2 → 模块 3 → 模块 4 → 模块 5 → 模块 6

## 总体优先级

| 优先级 | 目标 | 说明 |
|---|---|---|
| P0 | 训练正确性 | action mask、PPO old values、buffer mask、dual-clip 指标 |
| P0 | 环境目标闭环 | 塔、基地、队伍经济、objective rewards |
| P1 | 实验可复现 | CLI、evaluator、checkpoint、seed sweep、ablation |
| P1 | 采样效率 | 同步多环境 runner，落实 `num_envs` |
| P2 | Self-play/Curriculum | 让项目从“能训练”升级为“能对比” |
| P3 | LLM Planner MVP | 仅在 RL baseline 稳定后接入 |

---

## 模块 0：项目基线与开发护栏

### 概述

- 职责：建立后续修改的安全网，确保测试、lint、命令入口、配置加载可用。
- 前置依赖：无。
- 预计步骤数：5。

### Step 0.1：新增本地开发说明

- 操作：创建或更新 `docs/dev-setup.md`。
- 内容必须包含：
  - Python 版本：`3.10 <= version < 3.13`。
  - 安装命令：`pip install -e ".[dev,rl]"`。
  - 基础验证命令：
    - `python -m compileall hybrid_arena`
    - `pytest hybrid_arena/minimoba/tests -v`
    - `ruff check hybrid_arena`
    - `python hybrid_arena/scripts/benchmark_fps.py`
- 验证：`test -f docs/dev-setup.md`。

### Step 0.2：修正 pyproject 的 Python 版本边界

- 操作：在 `pyproject.toml` 中将 `requires-python = ">=3.10"` 改为：

```toml
requires-python = ">=3.10,<3.13"
```

- 理由：项目依赖的 RL/LLM 工具链通常对最新 Python 支持滞后；先锁到 3.10–3.12，减少安装不确定性。
- 验证：`python -m pip install -e ".[dev]"` 能解析项目元数据。

### Step 0.3：整理测试目录

- 操作：新增以下目录并放入空 `__init__.py`：
  - `hybrid_arena/algorithms/tests/__init__.py`
  - `hybrid_arena/training/tests/__init__.py`
  - `hybrid_arena/inference/tests/__init__.py`
- 验证：`python -m compileall hybrid_arena` 通过。

### Step 0.4：新增 smoke test 配置

- 操作：在 `pyproject.toml` 的 pytest markers 中增加：

```toml
"smoke: fast end-to-end smoke tests"
```

- 操作：创建 `hybrid_arena/minimoba/tests/test_smoke.py`，包含：
  - `test_env_smoke_2v2_20_steps`
  - 固定 seed，2v2，max_steps=20，随机合法动作，确认不抛异常。
- 验证：`pytest hybrid_arena/minimoba/tests/test_smoke.py -v` 通过。

### Step 0.5：建立 CI 工作流

- 操作：创建 `.github/workflows/ci.yml`。
- 内容：
  - Python 3.10、3.11、3.12 matrix。
  - 安装：`pip install -e ".[dev]"`。
  - 执行：`ruff check hybrid_arena`、`python -m compileall hybrid_arena`、`pytest hybrid_arena/minimoba/tests -v`。
- 验证：本地运行 `python -m compileall hybrid_arena` 与 `pytest hybrid_arena/minimoba/tests -v` 通过；推送后 GitHub Actions 通过。

### 验收标准

- [ ] 文档存在且命令可复制。
- [ ] Python 版本范围明确。
- [ ] 所有测试目录结构完整。
- [ ] smoke test 通过。
- [ ] CI 文件存在。

---

## 模块 1：动作编码与 Action Mask 语义修复

### 概述

- 职责：统一动作编码，修复 `(324,)` joint action mask 与网络 factorized head 的不一致问题。
- 前置依赖：模块 0。
- 预计步骤数：7。

### Step 1.1：新增动作编码工具

- 操作：创建 `hybrid_arena/minimoba/action_encoding.py`。
- 实现常量：

```python
N_MOVE = 9
N_SKILL = 4
N_TARGET = 9
N_ACTIONS = 324
```

- 实现函数：
  - `encode_action(move: int, skill: int, target: int) -> int`
  - `decode_action(index: int) -> tuple[int, int, int]`
  - `validate_action_components(move: int, skill: int, target: int) -> None`
- 验证：新增 `hybrid_arena/minimoba/tests/test_action_encoding.py`，覆盖 0、323、越界输入。
- 命令：`pytest hybrid_arena/minimoba/tests/test_action_encoding.py -v`。

### Step 1.2：把 GameState 中的 magic number 改为动作编码工具

- 操作：修改 `game_engine.py`：
  - 将 `mask[0 * 4 * 9 + 3 * 9 + 8]` 改为 `encode_action(0, 3, 8)`。
  - 将 `base = move_dir * 36` 相关逻辑改为 `encode_action(move_dir, skill, target)`。
- 验证：`pytest hybrid_arena/minimoba/tests/test_action_mask.py -v`，若文件尚不存在，先执行 Step 1.3。

### Step 1.3：新增 action mask 单元测试

- 操作：创建 `hybrid_arena/minimoba/tests/test_action_mask.py`。
- 测试用例：
  - `test_action_mask_shape_and_dtype`：shape 为 `(324,)`，合法值仅 0/1。
  - `test_no_attack_only_allows_target_8`：所有 move 下 skill=3 时仅 target=8 合法。
  - `test_stunned_hero_only_noop_action`：眩晕英雄只允许 `move=0, skill=3, target=8`。
  - `test_skill_cooldown_masks_skill_actions`：skill_1_cd > 0 时所有 skill=1 的 joint action 非法。
- 验证：`pytest hybrid_arena/minimoba/tests/test_action_mask.py -v` 通过。

### Step 1.4：修复 ActorCritic 的 joint logits

- 操作：修改 `hybrid_arena/algorithms/networks.py`。
- 在 `ActorCritic` 中新增私有方法：

```python
def _build_joint_logits(self, move_logits, skill_logits, target_logits):
    batch_size = move_logits.shape[0]
    joint = (
        move_logits[:, :, None, None]
        + skill_logits[:, None, :, None]
        + target_logits[:, None, None, :]
    )
    return joint.reshape(batch_size, 324)
```

- 修改 `get_action_and_value()`：
  - 使用 joint logits 构造 `Categorical(logits=joint_logits)`。
  - 如果 `action_mask is not None`，直接对 `(B,324)` mask 做 `masked_fill`。
  - 采样 flat action 后 decode 为 `(move, skill, target)`。
  - 如果传入 `action`，将 `(B,3)` encode 为 flat action，再计算 log_prob。
- 禁止继续使用 `action_mask[:, :9]`、`action_mask[:, 9:13]`、`action_mask[:, 13:22]`。
- 验证：`pytest hybrid_arena/algorithms/tests/test_joint_action_policy.py -v`，若文件尚不存在，先执行 Step 1.5。

### Step 1.5：新增 joint action policy 测试

- 操作：创建 `hybrid_arena/algorithms/tests/test_joint_action_policy.py`。
- 测试用例：
  - `test_policy_respects_single_valid_action_mask`：构造仅一个 index 合法的 mask，采样 20 次均返回该动作。
  - `test_log_prob_recompute_matches_sampled_action`：采样 action 后再传回 `get_action_and_value()`，log_prob shape 正确且有限。
  - `test_no_nan_when_mask_has_valid_action`：log_prob、entropy、value 均 finite。
- 验证：`pytest hybrid_arena/algorithms/tests/test_joint_action_policy.py -v` 通过。

### Step 1.6：让 RandomAgent 复用动作编码工具

- 操作：修改 `hybrid_arena/minimoba/agents/random_agent.py`。
- 将 flat index 到动作的解码逻辑改为调用 `decode_action(chosen)`。
- 验证：`pytest hybrid_arena/minimoba/tests/test_action_encoding.py hybrid_arena/minimoba/tests/test_action_mask.py -v`。

### Step 1.7：文档同步

- 操作：更新 README 的动作空间说明，补充：
  - 环境使用 joint action mask `(324,)`。
  - policy 内部用三个 head 生成 joint logits，并不是直接切片 mask。
- 验证：`grep -n "joint action mask" README.md` 有输出。

### 验收标准

- [ ] 任何地方不再把 `(324,)` mask 错切成 9/4/9 mask。
- [ ] joint action policy 在单合法动作 mask 下采样结果稳定。
- [ ] buffer/update 前暂未修复的 mask 问题在模块 2 继续处理。

---

## 模块 2：PPO / DualClipPPO 训练闭环修复

### 概述

- 职责：修复 PPO 的 old values、action masks、value clipping、dual-clip 指标，使训练日志可信。
- 前置依赖：模块 1。
- 预计步骤数：7。

### Step 2.1：RolloutBuffer 保存 action_mask

- 操作：修改 `hybrid_arena/training/buffer.py`。
- 新增成员：`self.action_masks: list[np.ndarray] = []`。
- 修改 `reset()` 清空 `action_masks`。
- 修改 `add()` 签名，增加 `action_mask_batch: np.ndarray`。
- 修改 `get_batch()` 返回：

```python
"action_masks": torch.tensor(mask.reshape(-1, 324), dtype=torch.int8, device=self.device)
```

- 验证：新增/更新 `hybrid_arena/training/tests/test_buffer.py`。
- 命令：`pytest hybrid_arena/training/tests/test_buffer.py -v`。

### Step 2.2：RolloutBuffer 返回 old_values

- 操作：继续修改 `buffer.py`。
- `get_batch()` 增加：

```python
"old_values": torch.tensor(val.reshape(-1), dtype=torch.float32, device=self.device)
```

- 测试：`test_buffer_returns_action_masks_and_old_values`。
- 验证：`pytest hybrid_arena/training/tests/test_buffer.py -v`。

### Step 2.3：Trainer 写入 action_mask

- 操作：修改 `hybrid_arena/training/trainer.py`。
- 在 `self.buffer.add(...)` 调用中传入：

```python
obs_np["action_mask"] 或 np.stack([obs[a]["action_mask"] for a in self.env.possible_agents])
```

- 注意：如果 `_stack_obs_np()` 当前不返回 action_mask，可新增返回字段。
- 验证：`pytest hybrid_arena/training/tests/test_buffer.py -v`。

### Step 2.4：PPO.update 接收 old_values

- 操作：修改 `hybrid_arena/algorithms/ppo/ppo.py`。
- `compute_loss()` 增加参数 `old_values: torch.Tensor`。
- `update()` 增加参数 `old_values: torch.Tensor`。
- minibatch 调用时传入 `old_values[mb_idx]`。
- 验证：新增 `hybrid_arena/algorithms/tests/test_ppo_loss.py` 中的 `test_ppo_update_accepts_old_values`。

### Step 2.5：修复 clipped value loss

- 操作：在 `PPO.compute_loss()` 中替换 value loss：

```python
value_pred_clipped = old_values + torch.clamp(
    new_values - old_values,
    -self.config.clip_eps,
    self.config.clip_eps,
)
value_losses = (new_values - returns) ** 2
value_losses_clipped = (value_pred_clipped - returns) ** 2
value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
```

- 验证：`pytest hybrid_arena/algorithms/tests/test_ppo_loss.py -v`。

### Step 2.6：Trainer.update 传入 action_masks 和 old_values

- 操作：修改 `trainer.py` 的 update 调用：

```python
info = self.algorithm.update(
    batch["obs"],
    batch["actions"],
    batch["log_probs"],
    batch["advantages"],
    batch["returns"],
    batch["action_masks"],
    batch["old_values"],
)
```

- 验证：创建 `hybrid_arena/training/tests/test_trainer_smoke.py`：
  - 使用 `PPOConfig(total_timesteps=32, num_steps=8, max_steps=20, map_size=16, team_size=2, device="cpu")`。
  - 调用 `Trainer(config, algo_type="ppo").train()`。
  - 确认返回 dict 包含 `episode_rewards`。
- 命令：`pytest hybrid_arena/training/tests/test_trainer_smoke.py -v -m smoke`。

### Step 2.7：修复 DualClipPPO 指标

- 操作：修改 `hybrid_arena/algorithms/ppo/ppo_dualclip.py`。
- 在求 mean 前计算 dual clip 是否触发：

```python
standard_surrogate = torch.min(surrogate1, surrogate2)
dual_clip_value = self.dual_clip_c * advantages
use_dual = (advantages < 0) & (standard_surrogate < dual_clip_value)
policy_objective = torch.where(use_dual, dual_clip_value, standard_surrogate)
policy_loss = -policy_objective.mean()
dual_frac = use_dual.float().mean().item()
```

- 同步修复 value loss，与 PPO 一致。
- 验证：`pytest hybrid_arena/algorithms/tests/test_ppo_loss.py -v`，新增 `test_dual_clip_fraction_is_finite`。

### 验收标准

- [ ] RolloutBuffer batch 包含 `action_masks` 和 `old_values`。
- [ ] PPO 更新阶段使用与 rollout 一致的 mask。
- [ ] value clipping 不再是无效表达式。
- [ ] DualClipPPO 指标 finite 且语义正确。
- [ ] CPU smoke training 可完成一个极小训练循环。

---

## 模块 3：MiniMOBA Objective Game 补完

### 概述

- 职责：补完塔、基地、队伍经济、objective rewards、终局条件，让环境支持高层策略。
- 前置依赖：模块 1。
- 预计步骤数：7。

### Step 3.1：新增结构物状态模型

- 操作：创建 `hybrid_arena/minimoba/objectives.py`。
- 实现：

```python
@dataclass
class StructureState:
    structure_id: str
    team: str
    structure_type: str  # "tower" | "base"
    x: int
    y: int
    max_hp: float
    hp: float
    attack_range: int = 4
    attack_damage: float = 30.0

    @property
    def alive(self) -> bool: ...
    @property
    def hp_ratio(self) -> float: ...
    def take_damage(self, amount: float) -> float: ...
```

- 验证：新增 `hybrid_arena/minimoba/tests/test_objectives.py::test_structure_state_damage`。

### Step 3.2：GameState 初始化 towers/bases

- 操作：修改 `game_engine.py`。
- 增加 `self.structures: dict[str, StructureState]`。
- 在 `reset()` 后根据地图中的 `RED_TOWER`、`BLUE_TOWER`、`RED_BASE`、`BLUE_BASE` 初始化结构物。
- 规则：
  - 每个 tower：`max_hp=1200`。
  - base：`max_hp=2000`。
- 验证：`test_game_state_initializes_structures`：2v2/4v4 reset 后红蓝双方 tower/base 数量正确。

### Step 3.3：实现攻击结构物

- 操作：修改 `_execute_attack()`。
- 当无合法 enemy hero target 且攻击/技能范围内存在敌方结构物时，允许攻击最近结构物。
- 仅允许 `skill_choice=0` 普攻攻击结构物；技能对结构物第一版不生效，避免奖励失控。
- 验证：`test_auto_attack_damages_enemy_tower_when_in_range`。

### Step 3.4：实现推塔奖励与队伍经济

- 操作：当 tower hp 降为 0 时：
  - 攻击者获得 `reward_config.tower`。
  - 敌方全队获得 `reward_config.tower_lost`。
  - 攻击方队伍经济增加 300。
  - 更新 `red_towers` / `blue_towers` 为存活 tower 数量。
- 击杀英雄时同步更新队伍经济：
  - red killer → `self.red_gold += gold_reward`
  - blue killer → `self.blue_gold += gold_reward`
- 验证：
  - `test_tower_destroy_updates_reward_and_counts`
  - `test_kill_updates_team_gold`

### Step 3.5：实现基地终局

- 操作：base 可在对应队伍 tower 全灭后被普攻伤害。
- 当 base hp <= 0：
  - `game_winner` 设为攻击方。
  - `is_game_over()` 返回 True。
- 若 max_steps 到达仍未爆基地，按以下顺序判胜：
  1. 存活 tower 数量多者。
  2. team gold 多者。
  3. kills 多者。
  4. draw。
- 验证：`test_base_destroy_ends_game_with_winner`、`test_max_steps_tiebreaker_uses_objectives`。

### Step 3.6：扩展 observation 与 info

- 操作：更新 `global_info` 保持 shape `(10,)` 不变，但替换 reserved 字段：
  - index 9 存放 objective advantage：`(red_tower_hp + red_base_hp - blue_tower_hp - blue_base_hp) / total_hp`，按阵营视角或全局固定视角二选一并写入注释。
- 更新 `infos[agent]` 增加：
  - `team_gold`
  - `enemy_gold`
  - `ally_tower_hp`
  - `enemy_tower_hp`
  - `ally_base_hp`
  - `enemy_base_hp`
- 验证：`test_observation_shapes` 仍通过；新增 `test_info_contains_objective_metrics`。

### Step 3.7：更新 reward 与 README

- 操作：更新 `RewardConfig` 注释与 README，说明：
  - kill/death/assist/damage/heal/tower/base/win/lose/time_penalty。
  - 当前版本中技能不伤害结构物。
- 验证：`grep -n "tower" README.md` 与 `grep -n "base" README.md` 均有对应说明。

### 验收标准

- [ ] 塔和基地有 runtime state。
- [ ] 推塔会改变奖励、经济和终局路径。
- [ ] 终局不再只靠 max_steps 与 kills。
- [ ] 高层策略有可观测 objective 信号。

---

## 模块 4：训练 CLI、评估器与实验复现

### 概述

- 职责：把“可调用 Trainer 类”升级为“可复现实验系统”。
- 前置依赖：模块 2、模块 3。
- 预计步骤数：7。

### Step 4.1：新增 checkpoint 工具

- 操作：创建 `hybrid_arena/training/checkpoint.py`。
- 实现：
  - `save_checkpoint(path, network, optimizer, config, global_step, metrics)`
  - `load_checkpoint(path, network=None, optimizer=None, map_location="cpu")`
- 保存内容：
  - `model_state_dict`
  - `optimizer_state_dict`
  - `config`
  - `global_step`
  - `metrics`
- 验证：`hybrid_arena/training/tests/test_checkpoint.py` 中保存后加载，参数张量一致。

### Step 4.2：Trainer 支持 checkpoint 与 metrics 输出

- 操作：修改 `trainer.py`。
- `Trainer.__init__()` 增加可选参数：
  - `checkpoint_dir: str | None = None`
  - `log_interval: int = 1000`
  - `save_interval: int = 50000`
- 每次保存：`checkpoints/{algo_type}_seed{seed}_step{global_timestep}.pt`。
- `train()` 返回 metrics dict，至少包含：
  - `episode_rewards`
  - `episode_lengths`
  - `global_timestep`
  - `fps`
  - `last_policy_loss`
  - `last_value_loss`
- 验证：`test_trainer_smoke_saves_checkpoint`。

### Step 4.3：新增 Evaluator

- 操作：创建 `hybrid_arena/training/evaluator.py`。
- 实现 `Evaluator.evaluate(policy, opponent_policy, n_episodes, seeds)`。
- 统计指标：
  - win_rate
  - draw_rate
  - avg_reward
  - avg_episode_length
  - avg_kills
  - avg_deaths
  - avg_towers_destroyed
  - avg_tower_hp_advantage
  - fps
- 验证：`hybrid_arena/training/tests/test_evaluator.py::test_evaluator_rule_vs_random_runs`。

### Step 4.4：新增 train CLI

- 操作：创建 `hybrid_arena/scripts/train.py`。
- argparse 参数：
  - `--algo {ppo,ppo_dualclip}`
  - `--seed int`
  - `--total-timesteps int`
  - `--num-steps int`
  - `--map-size int`
  - `--team-size int`
  - `--device {cpu,cuda}`
  - `--checkpoint-dir str`
- 命令示例：

```bash
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42 --total-timesteps 10000 --device cpu
```

- 验证：上述命令能完成 10k 或更小 smoke training。

### Step 4.5：新增 evaluate CLI

- 操作：创建 `hybrid_arena/scripts/evaluate.py`。
- argparse 参数：
  - `--checkpoint path`
  - `--opponent {random,rule_based,self}`
  - `--episodes int`
  - `--seed int`
  - `--output results/eval.json`
- 验证：

```bash
python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 5 --seed 42 --output results/eval_smoke.json
```

- 文件 `results/eval_smoke.json` 存在且包含 `win_rate`。

### Step 4.6：新增 ablation runner

- 操作：创建 `hybrid_arena/scripts/run_ablation.py`。
- 固定第一阶段矩阵：
  - `ppo` vs `ppo_dualclip`
  - seeds `[42, 123, 456]`
  - opponents `random`、`rule_based`
- 输出：
  - `results/ablation_raw.csv`
  - `results/ablation_summary.md`
- 验证：smoke 参数下可生成两个文件。

### Step 4.7：更新 README 的训练与评估章节

- 操作：替换 README 中仅 Python API 的训练示例，新增 CLI：

```bash
python -m hybrid_arena.scripts.train --algo ppo_dualclip --seed 42
python -m hybrid_arena.scripts.evaluate --checkpoint checkpoints/...pt --opponent rule_based --episodes 50
```

- 验证：`grep -n "scripts.train" README.md` 有输出。

### 验收标准

- [ ] 有 checkpoint 保存与加载。
- [ ] 有 evaluator。
- [ ] 有 train/evaluate/ablation CLI。
- [ ] 小规模 smoke train 与 smoke eval 可运行。
- [ ] README 命令与实际入口一致。

---

## 模块 5：采样效率、Self-play 与 Curriculum

### 概述

- 职责：落实 `num_envs`、实现历史策略池与课程学习，形成阶段 B 的实验亮点。
- 前置依赖：模块 4。
- 预计步骤数：6。

### Step 5.1：实现同步多环境 runner

- 操作：创建 `hybrid_arena/training/vector_runner.py`。
- 实现 `SyncParallelEnvRunner`：
  - 初始化 `num_envs` 个 `parallel_env`。
  - 每个 env 使用 `seed + env_idx`。
  - `reset()` 返回展平 batch：`num_envs * num_agents`。
  - `step(actions)` 接收展平 actions，再拆回每个 env。
- 验证：`hybrid_arena/training/tests/test_vector_runner.py::test_vector_runner_shapes`。

### Step 5.2：Trainer 使用 `config.num_envs`

- 操作：修改 `trainer.py`。
- 当 `config.num_envs > 1` 时使用 `SyncParallelEnvRunner`。
- Buffer 的 `num_agents` 改为 `num_envs * agents_per_env`。
- 日志中明确显示：`num_envs`、`agents_per_env`、`transitions_per_rollout`。
- 验证：`test_trainer_smoke_num_envs_2`。

### Step 5.3：新增 SelfPlayPool

- 操作：创建 `hybrid_arena/training/self_play.py`。
- 实现：
  - `SelfPlayPool(max_size: int)`
  - `add_checkpoint(path, metrics)`
  - `sample_opponent(strategy="recent_or_best")`
  - `list_opponents()`
- 第一版只保存 checkpoint path 与 win_rate，不做复杂 ELO。
- 验证：`hybrid_arena/training/tests/test_self_play.py`。

### Step 5.4：Evaluator 支持 self-play opponent

- 操作：修改 `evaluator.py`。
- opponent 可为：
  - `RandomAgent`
  - `RuleBasedAgent`
  - checkpoint policy
- 验证：`test_evaluator_checkpoint_opponent_contract` 使用临时 checkpoint 或 mock policy。

### Step 5.5：新增 CurriculumManager

- 操作：创建 `hybrid_arena/training/curriculum.py`。
- 从 `configs/default.yaml` 读取：
  - map_size
  - team_size
  - opponent
  - win_threshold
- 实现：
  - `current_level()`
  - `maybe_advance(metrics)`
  - `to_env_kwargs()`
- 规则：连续两次 eval win_rate >= threshold 才晋级。
- 验证：`hybrid_arena/training/tests/test_curriculum.py`。

### Step 5.6：训练日志输出阶段 B 结果表

- 操作：`run_ablation.py` 输出 markdown 表格：

| algo | seed | opponent | win_rate | avg_reward | avg_len | fps |
|---|---:|---|---:|---:|---:|---:|

- 验证：`test -f results/ablation_summary.md` 且包含 `win_rate`。

### 验收标准

- [ ] `config.num_envs` 实际生效。
- [ ] 支持 self-play checkpoint pool。
- [ ] 支持课程学习阶段切换。
- [ ] 有 PPO vs DualClipPPO 的可复现实验表。

---

## 模块 6：LLM 高层 Planner MVP

### 概述

- 职责：实现阶段 C 的最小可用版本。LLM 只做宏观策略，不直接执行微操。
- 前置依赖：模块 3、模块 4。
- 预计步骤数：6。

### Step 6.1：定义宏观动作集合

- 操作：创建 `hybrid_arena/inference/macro_actions.py`。
- 定义：

```python
MACRO_ACTIONS = [
    "group_mid",
    "push_nearest_tower",
    "retreat",
    "farm_safe",
    "protect_support",
    "force_teamfight",
    "split_push",
]
```

- 实现 `validate_macro_action(action: str) -> str`。
- 验证：`hybrid_arena/inference/tests/test_planner_contract.py::test_macro_action_validation`。

### Step 6.2：定义 PlannerState

- 操作：创建 `hybrid_arena/inference/planner_state.py`。
- 实现 dataclass：
  - `step`
  - `team`
  - `ally_summary`
  - `enemy_summary`
  - `objective_summary`
  - `score_summary`
- 实现 `summarize_game_state(game_state, team: str) -> PlannerState`。
- 验证：`test_summarize_game_state_returns_serializable_state`。

### Step 6.3：实现 RulePlanner 作为非 LLM 对照

- 操作：创建 `hybrid_arena/inference/rule_planner.py`。
- 策略：
  - 低血量比例高 → `retreat`
  - 敌方塔低血 → `push_nearest_tower`
  - 己方经济高且敌方可见 → `force_teamfight`
  - 默认 → `group_mid`
- 验证：`test_rule_planner_returns_valid_macro_action`。

### Step 6.4：实现 LLMPlanner 接口但默认离线可测

- 操作：创建 `hybrid_arena/inference/llm_planner.py`。
- 实现：
  - `BaseLLMClient` 协议：`generate(prompt: str) -> str`
  - `DummyLLMClient`：固定返回 `group_mid`，用于测试。
  - `LLMPlanner(client, model_name)`。
  - prompt 中明确要求只输出一个 macro action。
- 验证：`test_llm_planner_with_dummy_client`。
- 注意：不要在测试中调用真实外部 API。

### Step 6.5：实现 MacroActionAdapter

- 操作：创建 `hybrid_arena/inference/adapter.py`。
- 功能：将 macro action 转为 rule policy bias：
  - `retreat`：偏向远离敌人/回基地。
  - `push_nearest_tower`：偏向向敌方塔移动并普攻。
  - `force_teamfight`：偏向接近可见敌人并使用技能。
  - `group_mid`：偏向地图中心。
- 输出协议：`act(obs) -> np.ndarray`，兼容 env agent。
- 验证：`test_macro_action_adapter_outputs_valid_action`。

### Step 6.6：新增 planner demo 脚本

- 操作：创建 `hybrid_arena/scripts/play_planner.py`。
- 参数：
  - `--planner {rule,llm_dummy}`
  - `--team red`
  - `--max-steps 500`
  - `--render-mode rgb_array|human|none`
- 行为：每 10 步调用一次 planner，其他步复用上一 macro action。
- 验证：

```bash
python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 50 --render-mode none
```

### 验收标准

- [ ] PlannerState 可从 GameState 生成。
- [ ] RulePlanner 与 Dummy LLMPlanner 均可运行。
- [ ] Planner 不直接控制底层动作，而是通过 adapter 输出合法动作。
- [ ] 有 CLI demo。

---

## 模块 7：项目包装、文档与发布

### 概述

- 职责：把项目打磨成 GitHub 可展示、可复现、可面试讲解的状态。
- 前置依赖：模块 4，建议模块 5 至少完成 smoke。
- 预计步骤数：3。

### Step 7.1：重写 README 的结果章节

- 操作：在 README 中新增：
  - 当前实现状态表。
  - 训练命令。
  - 评估命令。
  - 已知限制。
  - 实验结果表占位。
- 验证：README 包含 `Known Limitations` 或 `已知限制`。

### Step 7.2：生成首版实验报告

- 操作：创建 `docs/experiment-report-v0.md`。
- 内容：
  - 环境设置。
  - PPO vs DualClipPPO smoke 实验。
  - RuleBased vs Random baseline。
  - 已知问题。
  - 后续实验矩阵。
- 验证：`test -f docs/experiment-report-v0.md`。

### Step 7.3：准备 GitHub release v0.2.0

- 操作：创建 `CHANGELOG.md`。
- 写入：
  - `v0.2.0 - training correctness and objective game milestone`。
  - Fixed / Added / Changed / Known Issues。
- 验证：`grep -n "v0.2.0" CHANGELOG.md` 有输出。

### 验收标准

- [ ] README 与实际 CLI 一致。
- [ ] 有首版实验报告。
- [ ] 有 CHANGELOG。
- [ ] GitHub 首页能直接展示项目价值与可复现命令。

---

## 不建议立即执行的事项

| 暂缓事项 | 原因 |
|---|---|
| 直接实现 GRPO/QLoRA | 当前 RL 训练闭环和 objective game 尚未稳定，GRPO 会放大问题 |
| 直接接 CrewAI 多 Agent | 演示效果好但不解决训练可信度问题 |
| 立即实现 MAPPO/QMIX/COMA | 先把 PPO/DualClipPPO 跑可信，再扩算法矩阵 |
| 引入 W&B 强绑定 | 第一版先本地 JSON/CSV/Markdown，避免外部账号依赖 |
| 做复杂 Web 前端 | Pygame/Streamlit demo 足够，核心应是训练与评估 |

## 第一里程碑 Definition of Done

当以下命令在本地全部通过，可视为下一步完善计划的第一里程碑完成：

```bash
python -m compileall hybrid_arena
ruff check hybrid_arena
pytest hybrid_arena/minimoba/tests -v
pytest hybrid_arena/algorithms/tests -v
pytest hybrid_arena/training/tests -v
python -m hybrid_arena.scripts.train --algo ppo --seed 42 --total-timesteps 512 --num-steps 32 --device cpu
python -m hybrid_arena.scripts.evaluate --opponent rule_based --episodes 3 --seed 42 --output results/eval_smoke.json
python -m hybrid_arena.scripts.play_planner --planner rule --max-steps 50 --render-mode none
```
