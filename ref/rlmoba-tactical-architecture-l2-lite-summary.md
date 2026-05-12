# RL/MOBA L2-lite 战术能力实现摘要

## 已实现文件

- `hybrid_arena/minimoba/tactical_runtime/memory.py`
  - 新增 `TacticalMemoryRecord` 与 sqlite3 `TacticalMemoryStore`。
  - 支持 `record`、`query`、`summarize_skill_outcomes`、`close`。
  - 创建索引 `(episode_id, agent_id, skill_id)` 与 `(skill_id, success)`。
- `hybrid_arena/minimoba/tactical_runtime/workspace.py`
  - 新增 `snapshot_annotations()`、`export_annotations()`、`import_annotations(rows)`。
  - 保留原有 annotation 查询、衰减、事件记录和 observation layer 行为。
- `hybrid_arena/minimoba/tactical_runtime/team_dispatcher.py`
  - 新增 `TeamDispatchResult` 与 `TeamTacticalDispatcher`。
  - 内部复用现有 `TacticalDispatcher`。
  - 对 objective/resource 坐标冲突做确定性处理：保留距离最近 agent，平局按 agent id；其他 agent 改为巡逻 fallback `{move: 0, skill: 0, target: 0}`。
- `hybrid_arena/minimoba/tactical_runtime/skill_stats.py`
  - 新增 `SkillOutcomeStats`、`update_stats`、`apply_stats_to_skill`。
  - 仅调整 skill prior 与 no-go traces，不做梯度训练。
  - prior clamp 到 `[0.05, 0.95]`，保留 skill 的 id/name/triggers/controller/provenance 等元数据。
- `hybrid_arena/minimoba/tactical_runtime/observation.py`
  - 新增 `build_augmented_observation(...)`。
  - 原 `local_map` 不变，新增 `local_map_with_pheromones`；默认 `local_map` 为 `(11, 11, 11)` 时增强后为 `(11, 11, 14)`。
- `hybrid_arena/minimoba/tactical_runtime/wrappers.py`
  - 新增 opt-in `PheromoneObservationAdapter`。
  - 未修改 `MiniMOBAEnv.observation_space()` 默认 contract。
- `hybrid_arena/minimoba/tactical_runtime/tactical_graph.py`
  - 新增轻量 dataclass `TacticalRelation`。
  - 支持 `annotations_to_relations`、`memory_to_relations`、`query_relations`。
- `hybrid_arena/minimoba/tactical_runtime/tests/`
  - 新增 memory、team dispatcher、skill stats、tactical graph、adapter 测试。
  - 扩展 workspace 与 observation 测试覆盖新接口。

## 验证结果

- `pytest hybrid_arena/minimoba/tactical_runtime/tests -v`
  - 84 passed
- `pytest hybrid_arena/minimoba/tests -v`
  - 60 passed, 1 skipped
- `ruff check hybrid_arena/minimoba/tactical_runtime hybrid_arena/minimoba`
  - All checks passed

## 限制

- Tactical memory 仍是 episode-level 存储层，没有接入每个 env tick 的默认路径。
- `PheromoneObservationAdapter` 仅作为显式 opt-in 包装器提供，不改变 MiniMOBA 默认 observation contract。
- Team conflict resolver 当前只识别 `farm_resources` 与 `push_objective` 对 resource/objective annotation 的坐标竞争。
- Tactical graph 是内存 dataclass 投影，不包含持久化图数据库、复杂图查询或跨 episode 推理。

## 下一轮候选

- 将 `TacticalMemoryStore` 接入 episode 结束后的批量 outcome 汇总，而不是 step 热路径。
- 为 team dispatcher 增加更多资源类型和角色优先级策略，但保持 deterministic。
- 在 demo 或离线评估脚本中增加 opt-in pheromone observation 示例。
- 将 tactical relation projection 用于调试面板或离线诊断报告。
