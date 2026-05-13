# AgentBench Agent Architecture L2-lite Summary

## 实现文件

- `hybrid_arena/skill_runtime/protocol.py`：本地 `SkillRuntimeMessage` / `SkillRuntimeError` envelope，含 message id、correlation id、source/target、JSON roundtrip 和 event/result adapter。
- `hybrid_arena/skill_runtime/memory.py`：SQLite `SkillMemoryStore`，支持 namespace/kind/tag/salience 查询、过期 decay、trace summary。
- `hybrid_arena/skill_runtime/workspace.py`：扩展 path glob / tag 查询、annotation snapshot、expired annotation prune。
- `hybrid_arena/skill_runtime/body_schema.py` 与 `dispatcher.py`：dispatch policy、path-specific annotation trigger、trace envelope metadata、affordance explainability。
- `hybrid_arena/skill_runtime/tool_registry.py`：静态本地 tool descriptor registry，暴露 deterministic controllers。
- `hybrid_arena/skill_runtime/adviser.py`：空 workspace、重复 escalation、低成功率、过期 annotation advisory。
- `hybrid_arena/scripts/skill_runtime_demo.py`：新增 `--list-tools`、`--explain-affordances`、`--event-json`。
- `hybrid_arena/services/api/app.py`：新增 `/skill-runtime/tools`、`/skill-runtime/advice`、`/skill-runtime/dispatch`。
- `hybrid_arena/skill_runtime/tests/` 与 `hybrid_arena/services/api/tests/`：覆盖协议、memory、dispatcher L2、tool registry、adviser、CLI 和 API。

## 验证

- `pytest hybrid_arena/skill_runtime/tests -v`：76 passed。
- `pytest hybrid_arena/services/api/tests -v`：6 passed。
- `python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once --list-tools`：通过。
- `python -m hybrid_arena.scripts.skill_runtime_demo --root . --db .skills/state.db --once --explain-affordances`：通过。
- `ruff check hybrid_arena/skill_runtime hybrid_arena/scripts hybrid_arena/services/api`：All checks passed。

## 限制

- 协议层仅为本地 envelope，不实现外部 ACP/A2A transport。
- Tool discovery 仅静态描述内置 deterministic controllers，不扫描网络、不动态导入外部代码。
- Adviser 只在 API/CLI 显式查询时返回建议，不做主动通知。

## 下一轮候选

- 将 trace-to-memory summary 接入更多真实 AgentBench scenario trace。
- 为 API 增加更细粒度的只读 trace 查询。
- 基于 review 结果决定是否把 dispatcher policy 参数外置到配置。
