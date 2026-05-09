"""Skill taxonomy for job-oriented AgentBench evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    label: str
    keywords: tuple[str, ...]


SKILL_TAXONOMY: dict[str, SkillDefinition] = {
    "python_backend": SkillDefinition(
        "python_backend",
        "Python 后端 / 脚本工程",
        ("python", "后端", "服务端", "脚本", "shell"),
    ),
    "http_api": SkillDefinition(
        "http_api",
        "HTTP / FastAPI / 接口",
        ("http", "api", "fastapi", "flask", "接口", "openapi"),
    ),
    "agent_workflow": SkillDefinition(
        "agent_workflow",
        "Agent 工作流",
        ("agent", "智能体", "工作流", "工具调用", "workflow"),
    ),
    "rag": SkillDefinition(
        "rag",
        "RAG / 知识库",
        ("rag", "知识库", "检索", "向量", "embedding"),
    ),
    "evaluation": SkillDefinition(
        "evaluation",
        "评测 / 报告 / 指标",
        ("评测", "测试题库", "指标", "报告", "benchmark", "eval"),
    ),
    "testing": SkillDefinition(
        "testing",
        "测试 / 自动化质量",
        ("测试", "自动化测试", "pytest", "回归", "质量"),
    ),
    "telecom_domain": SkillDefinition(
        "telecom_domain",
        "通信网络领域",
        ("通信", "网络", "5g", "核心网", "基站", "无线", "3gpp"),
    ),
    "deployment": SkillDefinition(
        "deployment",
        "部署 / Linux / Docker",
        ("部署", "linux", "docker", "k8s", "容器", "运维"),
    ),
    "communication": SkillDefinition(
        "communication",
        "沟通协作 / 交付",
        ("沟通", "协作", "跨团队", "文档", "交付"),
    ),
}


def normalize_skill_id(value: str) -> str | None:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in SKILL_TAXONOMY:
        return normalized
    for skill_id, definition in SKILL_TAXONOMY.items():
        if normalized == definition.label.lower().replace(" ", "_"):
            return skill_id
    return None
