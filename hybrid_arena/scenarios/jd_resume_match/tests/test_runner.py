from __future__ import annotations

from hybrid_arena.core.schema import TaskInput
from hybrid_arena.scenarios.jd_resume_match.runner import JDResumeMatchRunner


def test_jd_resume_match_runner_returns_trace_and_metrics() -> None:
    runner = JDResumeMatchRunner()
    task = TaskInput(
        task_id="jd-001",
        scenario="jd_resume_match",
        payload={
            "jd_text": "需要 Python、FastAPI、RAG、通信知识库经验。",
            "resume_profile": {
                "skills": ["python_backend", "http_api"],
                "evidence": {"python_backend": ["PPO trainer"], "http_api": ["API plan"]},
            },
        },
    )

    result = runner.run(task)

    assert result.scenario == "jd_resume_match"
    assert "rag" in result.output["missing_skills"]
    assert result.metrics["missing_skill_count"] >= 1
    assert [step.name for step in result.trace.steps] == ["extract_jd_requirements", "analyze_resume_gap"]
