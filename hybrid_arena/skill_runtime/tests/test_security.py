from __future__ import annotations

from hybrid_arena.skill_runtime.sample_skills import build_sample_skills
from hybrid_arena.skill_runtime.schema import Effect, Skill, SkillSignature
from hybrid_arena.skill_runtime.security import (
    RuntimePermissionPolicy,
    blocked_reason,
    filter_allowed_skills,
    is_skill_allowed,
    skill_effects,
)


def test_default_policy_blocks_write_capable_effects() -> None:
    skills = build_sample_skills()
    policy = RuntimePermissionPolicy()

    allowed = filter_allowed_skills(skills, policy)
    allowed_ids = {skill.id for skill in allowed}

    assert "inspect_workspace" in allowed_ids
    assert "write_summary" not in allowed_ids
    assert "run_shell" not in allowed_ids
    assert "fetch_url" not in allowed_ids
    assert "ask_llm" not in allowed_ids


def test_blocked_reason_names_denied_effects() -> None:
    write_skill = next(skill for skill in build_sample_skills() if skill.id == "write_summary")

    assert skill_effects(write_skill) == frozenset({Effect.READ_FS, Effect.WRITE_FS})
    assert is_skill_allowed(write_skill, RuntimePermissionPolicy()) is False
    assert "WRITE_FS" in blocked_reason(write_skill, RuntimePermissionPolicy())


def test_explicit_effect_allowlist_permits_specific_effect() -> None:
    write_skill = Skill(
        id="write_only",
        name="Write only",
        description="Writes a deterministic artifact.",
        signature=SkillSignature(
            inputs={"path": "str"},
            outputs={"path": "str"},
            effects=(Effect.WRITE_FS,),
        ),
    )
    shell_skill = Skill(
        id="shell",
        name="Shell",
        description="Runs a shell command.",
        signature=SkillSignature(
            inputs={"command": "str"},
            outputs={"stdout": "str"},
            effects=(Effect.RUN_SHELL,),
        ),
    )
    policy = RuntimePermissionPolicy(allowed_effects=frozenset({Effect.WRITE_FS}))

    assert is_skill_allowed(write_skill, policy) is True
    assert is_skill_allowed(shell_skill, policy) is False


def test_allow_write_effects_preserves_dev_demo_behavior() -> None:
    skills = build_sample_skills()
    policy = RuntimePermissionPolicy(allow_write_effects=True)

    assert {skill.id for skill in filter_allowed_skills(skills, policy)} == {skill.id for skill in skills}
