# AgentBench Benchmark Report

生成日期：2026-05-09

## 摘要

本报告记录 AgentBench 三个求职导向场景的本地离线 benchmark。所有样例均使用确定性规则或本地检索，不调用外部 LLM API。

| scenario | total | primary metrics |
|---|---:|---|
| jd_resume_match | 2 | skill_recall=1.0, evidence_coverage=1.0 |
| telecom_rag | 3 | recall_at_k=1.0, citation_coverage=1.0, unsupported_answer_rate=0.0 |
| ticket_triage | 5 | accuracy=1.0, macro_f1=1.0, unknown_rate=0.0 |

## JD 解析与简历差距分析

- 输入：`datasets/jd_samples/jd_cases.jsonl`
- 输出：`results/agentbench/jd_report.json`
- 方法：taxonomy + keyword evidence span。
- 指标：
  - `skill_recall=1.0`
  - `evidence_coverage=1.0`

该场景证明项目具备结构化 JD 解析、技能映射、证据链提取和简历差距分析能力。

## 通信知识库 RAG

- 输入：`datasets/telecom_docs/rag_eval_cases.jsonl`
- 知识库：`datasets/telecom_docs/mini_telecom_kb.jsonl`
- 输出：`results/agentbench/rag_report.json`
- 方法：token overlap + 中文字符/bigram + tag boost。
- 指标：
  - `recall_at_k=1.0`
  - `citation_coverage=1.0`
  - `unsupported_answer_rate=0.0`

该场景证明项目具备通信知识检索、引用式回答和幻觉控制的最小闭环。

## 网络工单分诊

- 输入：`datasets/ticket_samples/ticket_cases.jsonl`
- 输出：`results/agentbench/ticket_report.json`
- 方法：规则标签分类 + 置信度 + 排障建议模板。
- 指标：
  - `accuracy=1.0`
  - `macro_f1=1.0`
  - `unknown_rate=0.0`

该场景证明项目具备批量工单分诊、自动化测试评测和质量报告输出能力。

## 复现命令

```bash
python -m hybrid_arena.scripts.agentbench_run --scenario jd_resume_match --input datasets/jd_samples/jd_cases.jsonl --output results/agentbench/jd_report.json
python -m hybrid_arena.scripts.agentbench_run --scenario telecom_rag --input datasets/telecom_docs/rag_eval_cases.jsonl --output results/agentbench/rag_report.json
python -m hybrid_arena.scripts.agentbench_run --scenario ticket_triage --input datasets/ticket_samples/ticket_cases.jsonl --output results/agentbench/ticket_report.json
```
