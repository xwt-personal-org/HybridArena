from __future__ import annotations

from hybrid_arena.scenarios.jd_resume_match.taxonomy import SKILL_TAXONOMY, normalize_skill_id


def test_taxonomy_contains_job_oriented_skill_groups() -> None:
    assert "python_backend" in SKILL_TAXONOMY
    assert "agent_workflow" in SKILL_TAXONOMY
    assert "telecom_domain" in SKILL_TAXONOMY


def test_normalize_skill_id_rejects_unknown_skill() -> None:
    assert normalize_skill_id("Python Backend") == "python_backend"
    assert normalize_skill_id("unknown skill") is None
