# PettingZoo ParallelEnv 与 Action Mask — 实现参考

## 来源

- PettingZoo Parallel API：https://pettingzoo.farama.org/api/parallel/
- PettingZoo Action Masking：https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/

## 对应模块

- `docs/plan.md` 模块 1：动作编码与 Action Mask 语义修复
- `docs/plan.md` 模块 3：MiniMOBA Objective Game 补完

## 关键实现要点

1. `ParallelEnv.step(actions)` 接收所有当前 active agents 的动作字典。
2. `reset()` 返回 observations 和 infos 字典。
3. `step()` 返回 observations、rewards、terminations、truncations、infos 五个字典。
4. action mask 应和 action space 的语义严格一致。
5. HybridArena 当前环境使用 `MultiDiscrete([9, 4, 9])`，但 mask 是 joint mask `(324,)`，因此 policy 不能把它简单切成 `[9]`、`[4]`、`[9]`。

## 推荐实现

使用 324-way joint categorical：

```python
joint_logits = (
    move_logits[:, :, None, None]
    + skill_logits[:, None, :, None]
    + target_logits[:, None, None, :]
).reshape(batch_size, 324)

joint_logits = joint_logits.masked_fill(action_mask <= 0, -1e8)
dist = torch.distributions.Categorical(logits=joint_logits)
flat_action = dist.sample()
```

## 注意事项

- 如果所有 action 都被 mask 掉，必须在环境侧保证 noop action 合法。
- `skill=3` 表示 no attack 时，只允许 `target=8`。
- 更新阶段必须使用 rollout 时同一份 mask，否则 PPO ratio 不可信。
