# AgentBench 5 分钟演示脚本

## 0:00-0:30 项目定位

打开 README 或 Streamlit Overview。

讲法：

> 我把 HybridArena 从研究型多智能体项目重构成 AgentBench 平台。主线不是聊天 demo，而是任务运行、trace、评测和三个业务场景。

## 0:30-1:30 JD Match

打开 Streamlit 的 JD Match，粘贴一段 JD。

讲法：

> 这里做技能抽取和简历差距分析。输出是结构化 skill、evidence span、missing skills 和面试问题，不是自由文本。

## 1:30-2:30 Telecom RAG

提问：`AMF 的职责是什么`。

讲法：

> 这个场景把通信背景转成差异化优势。回答必须带 citation。没有证据时系统拒答，不伪造。

## 2:30-3:30 Ticket Triage

输入：`用户投诉 5G 基站覆盖差，室内频繁掉线。`

讲法：

> 这里展示通信工单分诊，输出分类、置信度、证据关键词和排障步骤。它也能批量评测 Macro-F1。

## 3:30-4:20 Trace

打开 Trace Contract 或接口返回 trace。

讲法：

> 三个场景共用 TaskRunResult 和 TaskTrace，所以 API、CLI、demo、benchmark 不需要各自写一套逻辑。

## 4:20-5:00 指标与总结

展示 `docs/agentbench-benchmark-report.md`。

讲法：

> 当前离线 benchmark 覆盖 JD skill recall、RAG recall/citation coverage、工单 accuracy/macro-F1。这个项目可以按岗位切换叙事：AI Agent 工程化、RAG、AI 评测、Python 后端、通信数智化。
