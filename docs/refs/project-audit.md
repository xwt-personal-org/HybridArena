# HybridArena 项目摸底与诊断报告

生成日期：2026-04-29

## 1. 分析对象

- 用户提供仓库地址：`https://github.com/w2030298-art/HybridArena.git`
- 当前公开访问结果：GitHub 页面不可直接读取，按上传的 `HybridArena-master.zip` 为准。
- 本次分析目录：`HybridArena-master/`
- 代码规模：`hybrid_arena/` 下约 3058 行 Python 代码。

## 2. 路线判定

判定为**复杂路线**。

理由：项目涉及不完全信息多智能体环境、PettingZoo ParallelEnv、PPO/DualClipPPO、self-play、课程学习、LLM Planner、GRPO/QLoRA 等算法与模型实现，不属于普通 CRUD 或成熟框架组装。

## 3. 当前项目结构

```text
hybrid_arena/
├── minimoba/          # MiniMOBA 多智能体环境
│   ├── env.py         # PettingZoo ParallelEnv
│   ├── game_engine.py # 战斗、视野、移动、奖励、终局逻辑
│   ├── hero.py        # 英雄配置和运行时状态
│   ├── map_generator.py
│   ├── reward_shaper.py
│   ├── renderer.py
│   ├── wrappers.py
│   ├── agents/        # RandomAgent / RuleBasedAgent
│   └── tests/         # API、环境、奖励测试
├── algorithms/
│   ├── networks.py    # MapEncoder / StateEncoder / ActorCritic
│   └── ppo/           # PPO / DualClipPPO / PPOConfig
├── training/
│   ├── buffer.py      # RolloutBuffer + GAE
│   └── trainer.py     # PPO 训练循环
├── inference/         # 目前仅 __init__.py
├── demo/              # 目前仅 __init__.py
├── configs/default.yaml
└── scripts/           # benchmark_fps.py / play_human.py
```

## 4. 已实现内容

### 4.1 环境层

- 已实现 PettingZoo Parallel API 风格的 `MiniMOBAEnv`。
- 已实现 4v4 / 2v2 可配置队伍规模。
- 已实现局部视野、迷雾、英雄属性、技能、冷却、死亡、复活。
- 已实现 RandomAgent、RuleBasedAgent。
- 已提供 Pygame 渲染与 FPS benchmark 脚本。

### 4.2 算法层

- 已实现 `MapEncoder`、`StateEncoder`、`ActorCritic`。
- 已实现 vanilla PPO 与 DualClipPPO 框架。
- 已实现 RolloutBuffer 与 GAE。
- 已有 `Trainer` 可执行单环境多智能体 PPO 训练。

### 4.3 文档层

- README 已清晰描述项目定位、阶段 A/B/C/D、动作空间、观测空间、硬件约束。
- `docs/` 中已有完整项目方案、招聘项目调研、RTX 4060 适配方案。

## 5. 静态验证结果

- `python -S -m compileall -q hybrid_arena`：通过，说明 Python 语法层面可编译。
- 当前容器的普通 `python`/`pip` site 初始化存在阻塞，未能在容器内完整运行 pytest；本计划中的验证命令应在本地正常 Python 3.10+ 环境或 Codex 环境执行。

## 6. 关键问题清单

### P0-1：Action mask 语义不一致，直接影响训练有效性

现状：

- 环境输出 `action_mask` 的形状为 `(324,)`，语义是 joint action mask：`move × skill × target`。
- `ActorCritic.get_action_and_value()` 将该 `(324,)` mask 错误切片为：
  - `action_mask[:, :9]` 作为 move mask
  - `action_mask[:, 9:13]` 作为 skill mask
  - `action_mask[:, 13:22]` 作为 target mask
- 这三个切片不是合法的 factorized mask，无法对应 `MultiDiscrete([9,4,9])` 的三个维度。
- `RolloutBuffer` 不保存 `action_mask`，`Trainer.update()` 中又传入 `None`，导致 PPO 更新阶段完全不使用动作掩码。

影响：

- 采样时可能采到错误分布下的动作。
- 训练时 log_prob 与 rollout 时的合法动作空间不一致。
- PPO ratio 失真，训练曲线即便上涨也不可靠。

建议优先修复：使用 324-way joint categorical logits 或实现严格的 conditional factorized policy。下一步计划采用 324-way joint categorical，改动最少且最安全。

### P0-2：PPO value clipping 实现无效

现状：

```python
value_pred_clipped = new_values + torch.clamp(
    new_values - new_values.detach(),
    -clip_eps,
    clip_eps,
)
```

数值上 `new_values - new_values.detach()` 为 0，因此 clipping 无实际作用。标准做法需要 rollout 时保存 old values，然后用 `old_values + clamp(new_values - old_values, ...)`。

影响：

- value loss 不是预期的 clipped value loss。
- 训练稳定性与日志指标不可信。

### P0-3：DualClipPPO 的 dual_clip_fraction 指标错误

现状：

- `policy_loss` 在求均值后变成标量，再与 `dual_clip_value` 向量比较。
- 指标不表示真实 dual-clip 触发比例。

影响：

- 算法可解释性下降。
- 实验报告中的 dual-clip 机制无法被正确审计。

### P0-4：塔、基地、经济差等核心 MOBA 目标未闭环

现状：

- `red_towers` / `blue_towers` 初始化为 2，但未在战斗中减少。
- `reward_config.tower` / `tower_lost` 已配置，但没有塔摧毁奖励触发点。
- `red_gold` / `blue_gold` 在 `global_info` 中使用，但击杀时只更新英雄个人 `gold`，未更新队伍经济。
- 终局基本依赖 `max_steps` 与击杀数，而不是推塔/基地目标。

影响：

- 当前环境更像“团战模拟器”，还不是“MOBA objective game”。
- 高层 LLM Planner 的“推塔/开团/撤退/分推”没有真实可操作目标。

### P1-1：`num_envs` 配置未使用

现状：

- `PPOConfig.num_envs = 4`，`configs/default.yaml` 也有 `n_envs`。
- `Trainer` 实际只创建一个 `parallel_env`，没有向量化多个环境副本。

影响：

- 采样吞吐与配置预期不一致。
- README/4060 适配方案中的并行环境假设未落地。

### P1-2：缺少训练 CLI、评估器、checkpoint 与实验记录

现状：

- README 中示例可用 Python API 启动训练，但 AGENTS.md 中写的 `training/train.py` 未实现。
- 没有 `scripts/train.py`、`scripts/evaluate.py`、`training/evaluator.py`。
- 没有 checkpoint 保存/加载、seed sweep、ablation 汇总。

影响：

- 项目难以复现实验结果。
- 简历项目缺少可展示的曲线、胜率表、模型文件与评估报告。

### P1-3：SingleAgentWrapper policy 调用协议不稳

现状：

- `SingleAgentWrapper.step()` 期待 `policy(self._obs[agent])`。
- 当前 `RuleBasedAgent` / `RandomAgent` 是 `.act(obs)` 协议，不可直接传入实例。

影响：

- README 中的 teammate/opponent policy 用法容易失败。

### P2-1：LLM Planner / demo / agent registry 仍为空目录

现状：

- `inference/`、`demo/`、`agents/` 基本只有 `__init__.py`。
- README 中阶段 C/D 尚未启动。

建议：不要立即上 GRPO。应先完成 P0/P1 训练有效性和环境目标闭环，再做 LLM Planner MVP。

## 7. 下一步总原则

1. **先保证训练正确，再扩功能**：action mask、PPO value loss、buffer mask、评估器是第一优先级。
2. **先让 MiniMOBA 有目标，再引入 LLM**：塔/基地/经济差/胜负条件必须闭环，否则 LLM 高层策略无处落地。
3. **先出可复现实验，再做 GRPO**：至少需要 PPO vs DualClipPPO vs RuleBased 的稳定评估矩阵。
4. **所有改动必须带测试**：本项目适合作为简历项目，不能只靠 README 叙事。
