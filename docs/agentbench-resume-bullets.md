# AgentBench 简历条目

## 中文版

- 将 HybridArena 从 RL/MOBA 研究项目重构为 AgentBench 平台，新增统一 `TaskInput/TaskRunResult/TaskTrace` contract，打通 FastAPI、CLI、Streamlit demo 与 SQLite run storage。
- 实现 JD 解析、通信知识库 RAG、网络工单分诊三个本地可运行场景，支持结构化输出、证据链、引用式回答、trace 回放和离线 benchmark。
- 建立 AgentBench 评测闭环：JD skill recall、RAG Recall@k/citation coverage、工单 accuracy/macro-F1，并生成 JSON/Markdown 报告；核心新主线测试 37 项通过。

## English Version

- Refactored HybridArena from an RL/MOBA research project into an AgentBench platform with shared `TaskInput`, `TaskRunResult`, and `TaskTrace` contracts across FastAPI, CLI, Streamlit, and SQLite storage.
- Built three offline runnable scenarios: JD/resume gap analysis, telecom RAG copilot, and network ticket triage, each with structured outputs, evidence/citations, traces, and evaluators.
- Added an evaluation loop covering skill recall, RAG Recall@k/citation coverage, ticket accuracy/macro-F1, and JSON/Markdown benchmark reports; 37 new AgentBench tests pass locally.
