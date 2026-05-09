# HybridArena 开发进度

## 当前状态

- 当前主线：AgentBench v3
- 最后更新：2026-05-09
- 状态：报告中的平台转向、三业务场景、API、CLI、demo 与面试文档已完成首版落地

## AgentBench v3 进度

### 阶段 A：平台转向与工程打底

- [x] A1：新增 `hybrid_arena/core/schema.py`
- [x] A2：新增 `TraceRecorder` 与 JSONL trace writer
- [x] A3：新增 SQLite `AgentBenchStore`
- [x] A4：新增 FastAPI 服务层
- [x] A5：新增场景注册表

### 阶段 B：JD 解析与简历差距 Agent

- [x] B1：新增 JD taxonomy 与样例数据
- [x] B2：实现 JD 技能抽取与 evidence span
- [x] B3：实现简历差距分析
- [x] B4：实现 runner 与离线评测

### 阶段 C：通信知识库 RAG Copilot

- [x] C1：新增通信知识库样例
- [x] C2：实现本地 retriever
- [x] C3：实现引用式回答
- [x] C4：实现 runner 与 RAG 评测

### 阶段 D：网络/工单分诊与评测台

- [x] D1：新增工单样例数据与标签
- [x] D2：实现工单分类与摘要
- [x] D3：实现排障建议
- [x] D4：实现 runner 与 macro-F1 评测

### 阶段 E：统一评测、Trace Viewer 与 Demo

- [x] E1：新增 `agentbench_run` CLI
- [x] E2：重写 Streamlit demo 为 AgentBench 面板
- [x] E3：新增 benchmark Markdown report

### 阶段 F：面试包装与质量收口

- [x] F1：重写 README 为 AgentBench 主线
- [x] F2：新增架构、演示脚本、简历条目
- [x] F3：更新 progress/issues/report/changelog
- [x] F4：全量验证与 ruff 收口

## 新主线局部验证

- `pytest hybrid_arena/core hybrid_arena/scenarios hybrid_arena/services/api hybrid_arena/scripts/tests -v`：37 passed
- `pytest hybrid_arena/minimoba/tests hybrid_arena/training/tests hybrid_arena/algorithms/tests -v`：96 passed, 1 skipped
- `pytest hybrid_arena/ -v`：180 passed, 1 skipped
- `python -m compileall hybrid_arena/demo/app.py`：通过
- `python -m compileall hybrid_arena`：通过
- `ruff check hybrid_arena`：All checks passed
- 三个 AgentBench CLI 报告已生成到 `results/agentbench/`

## Research 支线

MiniMOBA、PPO/DualClipPPO、MAPPO、QMIX、COMA、自博弈、课程学习和 LLM Planner 保留。F13 objective reward shaping 仍未解决 hard win 问题，暂不继续长训。
