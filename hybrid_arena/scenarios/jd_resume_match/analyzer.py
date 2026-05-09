"""Resume gap analysis for extracted JD requirements."""

from __future__ import annotations

from hybrid_arena.scenarios.jd_resume_match.extractor import SkillRequirement


def analyze_resume_gap(
    requirements: list[SkillRequirement],
    resume_profile: dict,
) -> dict:
    resume_skills = set(resume_profile.get("skills", []))
    evidence = resume_profile.get("evidence", {})
    required_skill_ids = [requirement.skill_id for requirement in requirements]

    matched_skills = {
        skill_id: list(evidence.get(skill_id, []))
        for skill_id in required_skill_ids
        if skill_id in resume_skills
    }
    missing_skills = [skill_id for skill_id in required_skill_ids if skill_id not in resume_skills]

    recommendations = [
        {
            "skill_id": skill_id,
            "action": _recommend_action(skill_id),
        }
        for skill_id in missing_skills
    ]
    interview_questions = [
        _question_for_skill(skill_id) for skill_id in required_skill_ids[:5]
    ]
    return {
        "required_skills": required_skill_ids,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations,
        "interview_questions": interview_questions,
    }


def _recommend_action(skill_id: str) -> str:
    actions = {
        "rag": "补充通信知识库 RAG 场景，展示检索、引用和幻觉控制指标。",
        "agent_workflow": "补充 Agent workflow trace，展示任务拆解、工具调用和失败回放。",
        "http_api": "补充 FastAPI/OpenAPI 服务入口和接口测试。",
        "testing": "补充接口回归、场景测试和 benchmark 报告。",
        "telecom_domain": "补充通信工单或 5G 知识库样例数据。",
    }
    return actions.get(skill_id, f"为 {skill_id} 增加可验证项目证据。")


def _question_for_skill(skill_id: str) -> str:
    return f"请说明你在项目中如何验证 {skill_id} 相关能力，并给出可复现指标。"
