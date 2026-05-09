"""Deterministic JD skill extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hybrid_arena.scenarios.jd_resume_match.taxonomy import SKILL_TAXONOMY


@dataclass(frozen=True)
class SkillRequirement:
    skill_id: str
    label: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def extract_jd_requirements(jd_text: str) -> list[SkillRequirement]:
    normalized_text = jd_text.lower()
    requirements: list[SkillRequirement] = []
    for skill_id, definition in SKILL_TAXONOMY.items():
        evidence = _first_keyword_evidence(jd_text, normalized_text, definition.keywords)
        if evidence is not None:
            requirements.append(
                SkillRequirement(
                    skill_id=skill_id,
                    label=definition.label,
                    evidence=evidence,
                )
            )
    return requirements


def _first_keyword_evidence(
    original_text: str,
    normalized_text: str,
    keywords: tuple[str, ...],
) -> str | None:
    for keyword in keywords:
        index = normalized_text.find(keyword.lower())
        if index >= 0:
            start = max(0, index - 8)
            end = min(len(original_text), index + len(keyword) + 8)
            return original_text[start:end]
    return None
