"""Streamlit demo for HybridArena AgentBench.

Run:
    streamlit run hybrid_arena/demo/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from hybrid_arena.core.schema import TaskInput
from hybrid_arena.scenarios.jd_resume_match.runner import JDResumeMatchRunner
from hybrid_arena.scenarios.telecom_rag.runner import TelecomRagRunner
from hybrid_arena.scenarios.ticket_triage.runner import TicketTriageRunner

st.set_page_config(page_title="HybridArena AgentBench", layout="wide")


def _load_resume_profile() -> dict:
    path = Path("datasets/jd_samples/resume_profile.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _render_result(result) -> None:
    metric_cols = st.columns(min(4, max(1, len(result.metrics))))
    for index, (key, value) in enumerate(result.metrics.items()):
        metric_cols[index % len(metric_cols)].metric(key, value)
    tab_output, tab_trace = st.tabs(["Output", "Trace"])
    with tab_output:
        st.json(result.output)
    with tab_trace:
        st.json(result.trace.to_dict())


st.title("HybridArena AgentBench")
st.caption("Agent workflow, RAG, ticket triage, trace, and evaluation in one local demo.")

with st.sidebar:
    st.header("场景")
    page = st.radio(
        "选择工作流",
        ["Overview", "JD Match", "Telecom RAG", "Ticket Triage", "Trace Contract"],
    )
    st.divider()
    st.markdown(
        """
        **本地可运行**

        - 不依赖外部 LLM API
        - 每次运行输出 trace
        - 与 FastAPI / CLI 共用 runner
        """
    )

if page == "Overview":
    st.subheader("平台主线")
    st.markdown(
        """
        HybridArena AgentBench 将原项目中的 planner、evaluator、trace 和 demo 能力迁移到求职导向的业务场景：
        JD 解析与简历差距分析、通信知识库 RAG、网络工单分诊。
        """
    )
    cols = st.columns(3)
    cols[0].metric("业务场景", 3)
    cols[1].metric("核心测试", "37+")
    cols[2].metric("外部 API 依赖", "0")
    st.subheader("接口入口")
    st.code(
        "\n".join(
            [
                "python -m hybrid_arena.services.api.app",
                "uvicorn hybrid_arena.services.api.app:app --reload",
                "python -m hybrid_arena.scripts.agentbench_run --scenario ticket_triage --input datasets/ticket_samples/ticket_cases.jsonl --output results/agentbench/ticket_report.json",
            ]
        ),
        language="bash",
    )

elif page == "JD Match":
    st.subheader("JD 解析与简历差距分析")
    jd_text = st.text_area(
        "JD 文本",
        "参与 AI Agent 工作流开发，要求熟悉 Python、HTTP API、FastAPI、RAG 和评测题库建设。",
        height=140,
    )
    if st.button("运行 JD Match", type="primary"):
        runner = JDResumeMatchRunner()
        result = runner.run(
            TaskInput(
                task_id="demo-jd",
                scenario="jd_resume_match",
                payload={"jd_text": jd_text, "resume_profile": _load_resume_profile()},
                metadata={"run_id": "demo-jd-run"},
            )
        )
        _render_result(result)

elif page == "Telecom RAG":
    st.subheader("通信知识库 RAG Copilot")
    question = st.text_input("问题", "AMF 的职责是什么")
    top_k = st.slider("Top K", 1, 5, 3)
    if st.button("运行 RAG", type="primary"):
        runner = TelecomRagRunner()
        result = runner.run(
            TaskInput(
                task_id="demo-rag",
                scenario="telecom_rag",
                payload={"question": question, "top_k": top_k},
                metadata={"run_id": "demo-rag-run"},
            )
        )
        _render_result(result)

elif page == "Ticket Triage":
    st.subheader("网络工单分诊与排障建议")
    ticket_text = st.text_area("工单内容", "用户投诉 5G 基站覆盖差，室内频繁掉线。", height=120)
    if st.button("运行分诊", type="primary"):
        runner = TicketTriageRunner()
        result = runner.run(
            TaskInput(
                task_id="demo-ticket",
                scenario="ticket_triage",
                payload={"ticket_text": ticket_text},
                metadata={"run_id": "demo-ticket-run"},
            )
        )
        _render_result(result)

else:
    st.subheader("Trace Contract")
    st.markdown(
        """
        所有场景返回同一种 `TaskRunResult`，包含业务输出、指标和步骤级 trace。
        这让 API、CLI、Streamlit demo 和离线评测可以复用同一套 runner。
        """
    )
    st.code(
        """
TaskRunResult(
    run_id="...",
    task_id="...",
    scenario="jd_resume_match | telecom_rag | ticket_triage",
    output={...},
    metrics={...},
    trace=TaskTrace(steps=[ToolCallRecord(...)]),
)
""".strip(),
        language="python",
    )
