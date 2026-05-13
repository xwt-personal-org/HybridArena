# RL/MOBA Tactical Architecture L2-lite Summary

## 实现文件

- `hybrid_arena/minimoba/tactical_runtime/memory.py`：SQLite `TacticalMemoryStore`，按 episode 记录 `TacticalMemoryRecord`，包含 state summary、action、reward delta、success 和 tags，不进入每 tick 默认热路径。
- `hybrid_arena/minimoba/tactical_runtime/workspace.py`：新增 annotation snapshot/export/import，保留原有空间查询、衰减、observation layer 行为。
- `hybrid_arena/minimoba/tactical_runtime/team_dispatcher.py`：新增 `TeamTacticalDispatcher` 和 deterministic conflict resolver，内部复用单 Agent `TacticalDispatcher`。
- `hybrid_arena/minimoba/tactical_runtime/skill_stats.py`：新增 reward_delta/success 驱动的 prior 与 no-go traces 更新，不引入 torch 训练。
- `hybrid_arena/minimoba/tactical_runtime/observation.py` 与 `wrappers.py`：新增 opt-in `local_map_with_pheromones` adapter，默认 `local_map` 仍为 `(11, 11, 11)`。
- `hybrid_arena/minimoba/tactical_runtime/tactical_graph.py`：新增轻量 `TacticalRelation` 投影和查询，不引入 graph DB。
- `hybrid_arena/minimoba/tactical_runtime/tests/`：覆盖 memory、team dispatcher、skill stats、observation adapter、relation export 和 workspace 扩展。

## 验证

- `pytest hybrid_arena/minimoba/tactical_runtime/tests -v`：85 passed。
- `pytest hybrid_arena/minimoba/tests -v`：60 passed, 1 skipped。
- `ruff check hybrid_arena/minimoba/tactical_runtime hybrid_arena/minimoba`：All checks passed。

## 限制

- Tactical memory 只提供 episode-level 存储层，尚未接入 episode 结束后的批量 outcome 汇总流程。
- Team conflict resolver 当前只处理 resource/objective 坐标竞争，策略保持 deterministic。
- Pheromone observation 仅通过显式 wrapper 启用，不改变 `MiniMOBAEnv.observation_space()` 默认 contract。
- Tactical graph 是 in-process dataclass 投影，不提供持久化图数据库或跨 episode 推理。

## 下一轮候选

- 将 `TacticalMemoryStore` 接入离线 evaluator 的 episode-end 汇总。
- 为 team dispatcher 增加角色优先级和更多目标类别，但继续保持 deterministic。
- 将 tactical relation projection 输出到 demo 或离线诊断报告。
