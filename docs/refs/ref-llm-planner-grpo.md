# LLM Planner 与 GRPO — 实现参考

## 来源

- LangGraph Overview：https://docs.langchain.com/oss/python/langgraph/overview
- Hugging Face TRL GRPOTrainer：https://huggingface.co/docs/trl/grpo_trainer

## 对应模块

- `docs/plan.md` 模块 6：LLM 高层 Planner MVP
- 后续阶段：GRPO / QLoRA 微调

## 当前建议

不要立即实现 GRPO。先完成：

1. RL baseline 可复现。
2. 环境 objective game 闭环。
3. PlannerState 可序列化。
4. RulePlanner 与 DummyLLMPlanner 可离线测试。
5. 收集 planner trace 数据。

## MVP 架构

```text
GameState
  -> PlannerState
  -> RulePlanner / LLMPlanner
  -> MacroAction
  -> MacroActionAdapter
  -> low-level action
```

## Macro Action 第一版集合

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

## GRPO 后续前置条件

- 至少有 `planner_traces.jsonl`，每条包含：
  - state summary
  - macro action
  - rollout outcome
  - scalar reward
- 有 reward function：win/loss、objective advantage、survival、tower damage。
- 有离线可复现训练脚本。
- 有显存策略：4060 Laptop 优先 Qwen2.5-1.5B 或 3B 4-bit。
