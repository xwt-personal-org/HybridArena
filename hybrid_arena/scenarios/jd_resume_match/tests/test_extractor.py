from __future__ import annotations

from hybrid_arena.scenarios.jd_resume_match.extractor import extract_jd_requirements


def test_extract_jd_requirements_finds_skills_with_evidence() -> None:
    text = "要求熟悉 Python 后端开发，了解 HTTP/FastAPI，做过 Agent 工作流和 RAG 评测。"

    requirements = extract_jd_requirements(text)
    skill_ids = {requirement.skill_id for requirement in requirements}

    assert {"python_backend", "http_api", "agent_workflow", "rag", "evaluation"} <= skill_ids
    assert all(requirement.evidence for requirement in requirements)
