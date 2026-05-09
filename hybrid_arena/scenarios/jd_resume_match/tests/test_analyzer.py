from __future__ import annotations

from hybrid_arena.scenarios.jd_resume_match.analyzer import analyze_resume_gap
from hybrid_arena.scenarios.jd_resume_match.extractor import extract_jd_requirements


def test_analyze_resume_gap_reports_missing_skills_and_recommendations() -> None:
    requirements = extract_jd_requirements("需要 Python、FastAPI、RAG、自动化测试经验。")
    resume_profile = {
        "skills": ["python_backend", "http_api"],
        "evidence": {
            "python_backend": ["HybridArena training CLI"],
            "http_api": ["FastAPI API layer"],
        },
    }

    report = analyze_resume_gap(requirements, resume_profile)

    assert "rag" in report["missing_skills"]
    assert "testing" in report["missing_skills"]
    assert report["matched_skills"]["python_backend"] == ["HybridArena training CLI"]
    assert any(item["skill_id"] == "rag" for item in report["recommendations"])
