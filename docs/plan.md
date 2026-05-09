# HybridArena AgentBench 系统化重构计划 v3

生成日期：2026-05-09

## 元信息

- 项目：HybridArena
- 新主线：HybridArena AgentBench
- 目标：按 `D:\下载\deep-research-report.md`，将项目从 RL/MOBA 研究主线重构为面向 AI Agent、AI 应用工程化、AI 评测、Python 后端与通信数智化岗位的可运行平台。
- 技术栈：Python 3.10+、Pydantic、FastAPI、SQLite、Pytest、Ruff、Streamlit、PyYAML。
- 保留策略：MiniMOBA、PPO、Evaluator、Planner、Trace 作为 `research` 能力保留，不删除现有环境与算法。
- 新增主线：`core` 平台抽象、`scenarios` 三个业务场景、`services/api` 后端接口、`demo` 展示页、`datasets` 样例数据、`docs` 面试交付物。
- 状态：执行中。

## 重构原则

1. 先交付可运行闭环，再扩展复杂能力。
2. 新业务场景先用 deterministic / rule-based / local mock 实现，LLM API 只作为可选适配层，测试不得依赖网络。
3. 每个场景都必须有：输入 schema、runner、trace、evaluator、样例数据、测试、API 或 demo 入口。
4. 旧 RL 训练不再作为求职主 showcase；仅保留在 README 的研究支线章节。
5. 不执行 300k-500k 长训，不新增 GPU 依赖。
6. 所有核心行为先写测试，再写实现。

## 阶段总览

| 阶段 | 目标 | 主要交付 |
|---|---|---|
| A | 平台转向与工程打底 | core task/run/trace/eval schema、SQLite storage、FastAPI app、基础 API 测试 |
| B | JD 解析与简历差距 Agent | JD 技能抽取、简历证据匹配、差距报告、离线评测 |
| C | 通信知识库 RAG Copilot | 文档切分、关键词检索、引用式回答、RAG 指标 |
| D | 网络/工单分诊与评测台 | 工单分类、摘要、排障建议、批处理、Macro-F1 |
| E | 统一评测、Trace Viewer 与 Demo | Streamlit 面板、trace 回放、benchmark report、导出样例 |
| F | 面试包装与质量收口 | README 重写、架构图、演示脚本、测试/ruff 全量验证 |

---

## 阶段 A：平台转向与工程打底

### A1：新增 AgentBench 核心 schema

- 文件：
  - 创建 `hybrid_arena/core/__init__.py`
  - 创建 `hybrid_arena/core/schema.py`
  - 创建 `hybrid_arena/core/tests/test_schema.py`
- 内容：
  - `TaskInput`：`task_id`、`scenario`、`payload`、`metadata`
  - `ToolCallRecord`：`name`、`input`、`output`、`latency_ms`、`success`
  - `TaskTrace`：`run_id`、`task_id`、`scenario`、`steps`、`metrics`
  - `TaskRunResult`：`run_id`、`task_id`、`scenario`、`output`、`metrics`、`trace`
  - `BenchmarkResult`：`scenario`、`total`、`metrics`、`cases`
- 验收：
  - schema 可 JSON round-trip。
  - `TaskRunResult` 必须内嵌完整 trace。
- 命令：
  - `pytest hybrid_arena/core/tests/test_schema.py -v`

### A2：新增统一 trace recorder

- 文件：
  - 创建 `hybrid_arena/core/traces.py`
  - 创建 `hybrid_arena/core/tests/test_traces.py`
- 内容：
  - `TraceRecorder.start_step(name, payload)`
  - `TraceRecorder.finish_step(output, success=True, metrics=None)`
  - `TraceRecorder.to_trace()`
  - `JsonlTraceWriter.write(trace)`
- 约束：
  - trace 记录 input、tool calls、output、metrics。
  - 不记录密钥、token、原始环境变量。
- 命令：
  - `pytest hybrid_arena/core/tests/test_traces.py -v`

### A3：新增 SQLite storage

- 文件：
  - 创建 `hybrid_arena/core/storage.py`
  - 创建 `hybrid_arena/core/tests/test_storage.py`
- 内容：
  - `AgentBenchStore(db_path)`
  - `init_schema()`
  - `save_run(result)`
  - `get_run(run_id)`
  - `list_runs(scenario=None, limit=50)`
- 约束：
  - 使用标准库 `sqlite3`，不引入 SQLAlchemy 首版依赖。
  - `payload`、`output`、`metrics`、`trace` 以 JSON text 保存。
- 命令：
  - `pytest hybrid_arena/core/tests/test_storage.py -v`

### A4：新增 FastAPI 服务层

- 文件：
  - 修改 `pyproject.toml`：新增 optional dependency `app = ["fastapi>=0.110", "uvicorn>=0.29", "streamlit>=1.35"]`
  - 创建 `hybrid_arena/services/__init__.py`
  - 创建 `hybrid_arena/services/api/__init__.py`
  - 创建 `hybrid_arena/services/api/app.py`
  - 创建 `hybrid_arena/services/api/tests/test_app.py`
- API：
  - `GET /health`
  - `GET /scenarios`
  - `POST /tasks/run`
  - `GET /runs/{run_id}`
  - `GET /runs`
- 验收：
  - 无外部 LLM key 时仍可运行。
  - OpenAPI 可生成。
- 命令：
  - `pip install -e ".[dev,app]"`
  - `pytest hybrid_arena/services/api/tests/test_app.py -v`

### A5：新增场景注册表

- 文件：
  - 创建 `hybrid_arena/scenarios/__init__.py`
  - 创建 `hybrid_arena/scenarios/registry.py`
  - 创建 `hybrid_arena/scenarios/tests/test_registry.py`
- 内容：
  - `ScenarioRunner` protocol。
  - `get_runner(scenario_name)`。
  - `list_scenarios()`。
- 首版场景名：
  - `jd_resume_match`
  - `telecom_rag`
  - `ticket_triage`
- 命令：
  - `pytest hybrid_arena/scenarios/tests/test_registry.py -v`

---

## 阶段 B：JD 解析与简历差距 Agent

### B1：新增技能 taxonomy 与样例数据

- 文件：
  - 创建 `datasets/jd_samples/jd_cases.jsonl`
  - 创建 `datasets/jd_samples/resume_profile.json`
  - 创建 `hybrid_arena/scenarios/jd_resume_match/taxonomy.py`
  - 创建 `hybrid_arena/scenarios/jd_resume_match/tests/test_taxonomy.py`
- 技能类别：
  - `python_backend`
  - `http_api`
  - `agent_workflow`
  - `rag`
  - `evaluation`
  - `testing`
  - `telecom_domain`
  - `deployment`
  - `communication`
- 命令：
  - `pytest hybrid_arena/scenarios/jd_resume_match/tests/test_taxonomy.py -v`

### B2：实现 JD 技能抽取

- 文件：
  - 创建 `hybrid_arena/scenarios/jd_resume_match/extractor.py`
  - 创建 `hybrid_arena/scenarios/jd_resume_match/tests/test_extractor.py`
- 行为：
  - 输入 JD 文本。
  - 输出结构化 skill requirements。
  - 使用规则词表与中文/英文关键词匹配。
  - 返回 evidence span。
- 命令：
  - `pytest hybrid_arena/scenarios/jd_resume_match/tests/test_extractor.py -v`

### B3：实现简历差距分析

- 文件：
  - 创建 `hybrid_arena/scenarios/jd_resume_match/analyzer.py`
  - 创建 `hybrid_arena/scenarios/jd_resume_match/tests/test_analyzer.py`
- 行为：
  - 输入 JD 技能要求和简历 profile。
  - 输出 `matched_skills`、`missing_skills`、`recommendations`、`interview_questions`。
  - 推荐项必须绑定缺失技能和项目证据。
- 命令：
  - `pytest hybrid_arena/scenarios/jd_resume_match/tests/test_analyzer.py -v`

### B4：实现 JD 场景 runner 与评测

- 文件：
  - 创建 `hybrid_arena/scenarios/jd_resume_match/runner.py`
  - 创建 `hybrid_arena/scenarios/jd_resume_match/evaluator.py`
  - 创建 `hybrid_arena/scenarios/jd_resume_match/tests/test_runner.py`
  - 创建 `hybrid_arena/scenarios/jd_resume_match/tests/test_evaluator.py`
- 指标：
  - `skill_recall`
  - `evidence_coverage`
  - `missing_skill_count`
- 命令：
  - `pytest hybrid_arena/scenarios/jd_resume_match/tests -v`

---

## 阶段 C：通信知识库 RAG Copilot

### C1：新增通信文档样例

- 文件：
  - 创建 `datasets/telecom_docs/mini_telecom_kb.jsonl`
  - 创建 `hybrid_arena/scenarios/telecom_rag/corpus.py`
  - 创建 `hybrid_arena/scenarios/telecom_rag/tests/test_corpus.py`
- 内容：
  - 至少包含 5 条通信知识 chunk。
  - 字段：`doc_id`、`title`、`source`、`text`、`tags`。
- 命令：
  - `pytest hybrid_arena/scenarios/telecom_rag/tests/test_corpus.py -v`

### C2：实现本地检索器

- 文件：
  - 创建 `hybrid_arena/scenarios/telecom_rag/retriever.py`
  - 创建 `hybrid_arena/scenarios/telecom_rag/tests/test_retriever.py`
- 行为：
  - 无 embedding 依赖。
  - 使用 token overlap + tag boost。
  - 返回 top-k chunks 和 score。
- 命令：
  - `pytest hybrid_arena/scenarios/telecom_rag/tests/test_retriever.py -v`

### C3：实现引用式回答生成

- 文件：
  - 创建 `hybrid_arena/scenarios/telecom_rag/generator.py`
  - 创建 `hybrid_arena/scenarios/telecom_rag/tests/test_generator.py`
- 行为：
  - 回答必须带 citations。
  - 未检索到证据时返回“不足以回答”，不编造。
- 命令：
  - `pytest hybrid_arena/scenarios/telecom_rag/tests/test_generator.py -v`

### C4：实现 RAG runner 与评测

- 文件：
  - 创建 `hybrid_arena/scenarios/telecom_rag/runner.py`
  - 创建 `hybrid_arena/scenarios/telecom_rag/evaluator.py`
  - 创建 `hybrid_arena/scenarios/telecom_rag/tests/test_runner.py`
  - 创建 `hybrid_arena/scenarios/telecom_rag/tests/test_evaluator.py`
- 指标：
  - `recall_at_k`
  - `citation_coverage`
  - `unsupported_answer_rate`
- 命令：
  - `pytest hybrid_arena/scenarios/telecom_rag/tests -v`

---

## 阶段 D：网络/工单分诊与评测台

### D1：新增工单样例数据

- 文件：
  - 创建 `datasets/ticket_samples/ticket_cases.jsonl`
  - 创建 `hybrid_arena/scenarios/ticket_triage/labels.py`
  - 创建 `hybrid_arena/scenarios/ticket_triage/tests/test_labels.py`
- 标签：
  - `radio_access`
  - `core_network`
  - `transport`
  - `device`
  - `billing`
  - `unknown`
- 命令：
  - `pytest hybrid_arena/scenarios/ticket_triage/tests/test_labels.py -v`

### D2：实现工单分类与摘要

- 文件：
  - 创建 `hybrid_arena/scenarios/ticket_triage/classifier.py`
  - 创建 `hybrid_arena/scenarios/ticket_triage/tests/test_classifier.py`
- 行为：
  - 输出 label、confidence、evidence_keywords。
  - 生成 1 句摘要。
- 命令：
  - `pytest hybrid_arena/scenarios/ticket_triage/tests/test_classifier.py -v`

### D3：实现排障建议生成

- 文件：
  - 创建 `hybrid_arena/scenarios/ticket_triage/recommender.py`
  - 创建 `hybrid_arena/scenarios/ticket_triage/tests/test_recommender.py`
- 行为：
  - 每个标签输出 3-5 条排障步骤。
  - unknown 标签提示补充信息，不伪造确定结论。
- 命令：
  - `pytest hybrid_arena/scenarios/ticket_triage/tests/test_recommender.py -v`

### D4：实现批处理 runner 与评测

- 文件：
  - 创建 `hybrid_arena/scenarios/ticket_triage/runner.py`
  - 创建 `hybrid_arena/scenarios/ticket_triage/evaluator.py`
  - 创建 `hybrid_arena/scenarios/ticket_triage/tests/test_runner.py`
  - 创建 `hybrid_arena/scenarios/ticket_triage/tests/test_evaluator.py`
- 指标：
  - `accuracy`
  - `macro_f1`
  - `unknown_rate`
- 命令：
  - `pytest hybrid_arena/scenarios/ticket_triage/tests -v`

---

## 阶段 E：统一评测、Trace Viewer 与 Demo

### E1：新增统一 benchmark CLI

- 文件：
  - 创建 `hybrid_arena/scripts/agentbench_run.py`
  - 创建 `hybrid_arena/scripts/tests/test_agentbench_run.py`
- CLI：
  - `python -m hybrid_arena.scripts.agentbench_run --scenario jd_resume_match --input datasets/jd_samples/jd_cases.jsonl --output results/agentbench/jd_report.json`
  - `python -m hybrid_arena.scripts.agentbench_run --scenario telecom_rag --input datasets/telecom_docs/mini_telecom_kb.jsonl --output results/agentbench/rag_report.json`
  - `python -m hybrid_arena.scripts.agentbench_run --scenario ticket_triage --input datasets/ticket_samples/ticket_cases.jsonl --output results/agentbench/ticket_report.json`
- 命令：
  - `pytest hybrid_arena/scripts/tests/test_agentbench_run.py -v`

### E2：重写 Streamlit demo

- 文件：
  - 修改 `hybrid_arena/demo/app.py`
- 页面：
  - Overview
  - JD Match
  - Telecom RAG
  - Ticket Triage
  - Trace Viewer
  - Benchmarks
- 命令：
  - `python -m compileall hybrid_arena/demo/app.py`

### E3：新增 benchmark report 生成

- 文件：
  - 创建 `hybrid_arena/core/reporting.py`
  - 创建 `hybrid_arena/core/tests/test_reporting.py`
  - 创建 `docs/agentbench-benchmark-report.md`
- 内容：
  - 汇总三场景指标。
  - 输出错误样例和 trace 链接。
- 命令：
  - `pytest hybrid_arena/core/tests/test_reporting.py -v`

---

## 阶段 F：面试包装与质量收口

### F1：重写 README 首页

- 文件：
  - 修改 `README.md`
- 内容：
  - 新定位：AgentBench 平台。
  - 三场景架构图。
  - 快速启动。
  - API 列表。
  - Demo 流程。
  - 研究支线：MiniMOBA / RL。
- 验收：
  - README 首屏不再把 MOBA/RL 作为唯一主线。

### F2：新增面试材料

- 文件：
  - 创建 `docs/agentbench-demo-script.md`
  - 创建 `docs/agentbench-architecture.md`
  - 创建 `docs/agentbench-resume-bullets.md`
- 内容：
  - 5 分钟演示脚本。
  - 架构说明。
  - 简历项目条目。

### F3：更新进度、问题与变更记录

- 文件：
  - 修改 `docs/progress.md`
  - 修改 `docs/issues.md`
  - 修改 `docs/report.md`
  - 修改 `CHANGELOG.md`
- 内容：
  - 标记 AgentBench v3 阶段完成情况。
  - 保留 F13 未解决问题，但移到 research 支线。

### F4：全量验证

- 命令：
  - `python -m compileall hybrid_arena`
  - `pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v`
  - `pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v`
  - `ruff check hybrid_arena`

## Definition of Done

- [ ] `docs/inbox/plan.md` 已删除。
- [ ] `docs/plan.md` 切换为 AgentBench v3。
- [ ] 三个业务场景均有 runner、trace、evaluator、测试和样例数据。
- [ ] FastAPI 服务可运行并覆盖核心接口测试。
- [ ] Streamlit demo 从 RL 对战页改为 AgentBench 展示页。
- [ ] README、架构、演示脚本、benchmark report 可直接用于面试展示。
- [ ] 核心测试、旧模块回归测试、ruff 全部通过，或明确记录不可通过原因。
