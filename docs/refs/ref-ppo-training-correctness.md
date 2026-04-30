# PPO / DualClipPPO 训练正确性 — 实现参考

## 来源

- CleanRL PPO 文档：https://docs.cleanrl.dev/rl-algorithms/ppo/
- PPO implementation details 参考：https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

## 对应模块

- `docs/plan.md` 模块 2：PPO / DualClipPPO 训练闭环修复
- `docs/plan.md` 模块 4：训练 CLI、评估器与实验复现

## 关键实现要点

### 1. Rollout 数据必须完整

PPO update 至少需要：

```python
obs
actions
old_log_probs
old_values
action_masks
advantages
returns
```

HybridArena 当前 buffer 缺少 `action_masks` 和 `old_values`。

### 2. Clipped value loss 必须使用 old values

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

### 3. DualClipPPO 指标先计算后 reduce

```python
standard_surrogate = torch.min(surrogate1, surrogate2)
dual_clip_value = dual_clip_c * advantages
use_dual = (advantages < 0) & (standard_surrogate < dual_clip_value)
policy_objective = torch.where(use_dual, dual_clip_value, standard_surrogate)
policy_loss = -policy_objective.mean()
dual_clip_fraction = use_dual.float().mean().item()
```

## 注意事项

- `old_log_probs` 必须来自采样策略。
- update 阶段 action mask 必须与采样阶段一致。
- smoke training 不等于算法有效，只能证明端到端无异常。
- 正式结果需要至少 3 个 seeds 和固定对手评估。
