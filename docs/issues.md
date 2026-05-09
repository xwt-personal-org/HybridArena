# HybridArena 问题记录

## 当前执行阻塞

暂无阻塞；待最终全量验证与 ruff 收口。

## AgentBench v3 已发现并修复的问题

### ISSUE-A3：pytest 新测试目录模块名冲突

- 严重级别：P1
- 现象：组合运行 `hybrid_arena/scenarios` 时，多个目录下的 `test_runner.py` / `test_evaluator.py` 被 pytest 当作同名顶层模块导入。
- 修复：为 `hybrid_arena/core/tests`、`hybrid_arena/scenarios/*/tests`、`hybrid_arena/services/api/tests` 增加 `__init__.py`。
- 状态：已修复，组合测试 37 passed。

### ISSUE-C2：中文 RAG 检索无法召回“网络丢包排查”

- 严重级别：P1
- 现象：`telecom_rag` 对中文长句只提取整段 token，导致 `packet-loss` chunk 未召回。
- 修复：retriever 增加中文单字与 bigram token，保留英文 token 与 tag boost。
- 状态：已修复，`telecom_rag` benchmark `recall_at_k=1.0`。

### ISSUE-A4：FastAPI 测试依赖缺失

- 严重级别：P2
- 现象：`fastapi.testclient` 需要 `httpx`，首次 `app` extras 未声明。
- 修复：`pyproject.toml` 的 `app` extras 增加 `httpx>=0.27`。
- 状态：已修复，API 测试 3 passed。

## Research 支线遗留问题

### ISSUE-F13：objective reward shaping 未产生 hard win

- 严重级别：P2
- 现象：tower_damage 提升，但 `hard_win_rate=0.0`、`base_exposed_rate=0.0`、`avg_base_damage=0.0`。
- 当前处理：从求职主线移出，保留为 research 支线问题。
- 建议：只有当 scripted objective policy 能稳定触达 base objective 后，才考虑 300k-500k 长训。
