"""Body-schema: applicability filter and affordance ranking for skills."""

from __future__ import annotations

import fnmatch
import json
import re

from hybrid_arena.skill_runtime.schema import Skill, Trigger, WorkspaceEvent
from hybrid_arena.skill_runtime.workspace import Workspace


class BodySchema:
    """Maintains the set of currently applicable skills and ranks them.

    The body-schema is updated reactively when a :class:`WorkspaceEvent`
    arrives.  Skills whose *preconditions* are all satisfied by the workspace
    annotations are considered **applicable** and ranked by
    ``applicability × salience``.

    Args:
        skills: Full list of registered skills.
        workspace: Backing workspace instance.
    """

    def __init__(self, skills: list[Skill], workspace: Workspace) -> None:
        self._skills = list(skills)
        self._workspace = workspace

    def update(self, event: WorkspaceEvent) -> None:
        """React to a workspace event.

        Currently records the event for bookkeeping; the L0 prototype does
        not perform any state mutation beyond logging.

        Args:
            event: The workspace event to process.
        """
        self._workspace.record_event(
            kind=event.kind,
            path=event.path,
            payload=dict(event.payload),
        )

    def _is_applicable(self, skill: Skill) -> bool:
        """Return ``True`` if all preconditions of *skill* are satisfied.

        An empty *preconditions* tuple means the skill is unconditionally
        applicable.
        """
        if not skill.preconditions:
            return True
        # Each precondition is checked as a tag on any path in the workspace.
        for cond in skill.preconditions:
            paths = self._workspace.query_paths(any_tag=cond)
            if not paths:
                return False
        return True

    def _missing_preconditions(self, skill: Skill) -> list[str]:
        missing: list[str] = []
        for cond in skill.preconditions:
            if not self._workspace.query_paths(any_tag=cond):
                missing.append(cond)
        return missing

    def _annotation_event_path_match(
        self,
        spec: str,
        event: WorkspaceEvent | None,
    ) -> bool:
        if event is None or not event.path:
            return False
        annotation = self._workspace.get_annotation(event.path)
        if annotation is None:
            return False
        return spec in annotation.tags or annotation.status == spec

    def _trigger_diagnostic(
        self,
        trigger: Trigger,
        event: WorkspaceEvent | None,
    ) -> dict:
        event_path_match = False
        global_match = False
        matched = False
        if event is not None:
            if trigger.kind == "glob":
                matched = fnmatch.fnmatch(event.path, trigger.spec)
                event_path_match = matched
            elif trigger.kind == "regex":
                matched = bool(re.fullmatch(trigger.spec, event.path))
                event_path_match = matched
            elif trigger.kind == "annotation":
                event_path_match = self._annotation_event_path_match(
                    trigger.spec, event
                )
                global_match = bool(
                    self._workspace.query_paths(any_tag=trigger.spec)
                    or self._workspace.query_paths(status=trigger.spec)
                )
                matched = event_path_match or global_match
        return {
            "kind": trigger.kind,
            "spec": trigger.spec,
            "salience": trigger.salience,
            "matched": matched,
            "event_path_match": event_path_match,
            "global_match": global_match,
        }

    def current_affordances(self, top_k: int = 8) -> list[Skill]:
        """Return up to *top_k* applicable skills, ranked by salience.

        Ranking formula: ``salience`` (all skills are applicable in L0 when
        preconditions are empty).

        Args:
            top_k: Maximum number of skills to return.
        """
        applicable = [s for s in self._skills if self._is_applicable(s)]
        applicable.sort(key=lambda s: s.salience, reverse=True)
        return applicable[:top_k]

    def to_prompt_summary(self) -> str:
        """Return a human-readable summary of current affordances.

        Suitable for inclusion in an LLM prompt context.
        """
        affordances = self.current_affordances()
        if not affordances:
            return "No applicable skills."
        lines = ["Current affordances:"]
        for skill in affordances:
            trigger_desc = ", ".join(
                f"{t.kind}:{t.spec}" for t in skill.triggers
            )
            lines.append(
                f"  - {skill.name} (id={skill.id}, salience={skill.salience}, "
                f"triggers=[{trigger_desc}])"
            )
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of the current state."""
        affordances = self.current_affordances()
        return {
            "skill_count": len(self._skills),
            "affordance_count": len(affordances),
            "affordances": [
                {
                    "id": s.id,
                    "name": s.name,
                    "salience": s.salience,
                    "triggers": [
                        {"kind": t.kind, "spec": t.spec, "salience": t.salience}
                        for t in s.triggers
                    ],
                }
                for s in affordances
            ],
        }

    def explain_affordances(
        self,
        event: WorkspaceEvent | None = None,
        top_k: int = 8,
    ) -> list[dict]:
        """Return JSON-serializable diagnostics for current affordances."""
        ranked = sorted(self._skills, key=lambda s: s.salience, reverse=True)[:top_k]
        event_context = (
            {
                "kind": event.kind,
                "path": event.path,
                "payload": dict(event.payload),
                "created_at": event.created_at,
            }
            if event is not None
            else None
        )
        return [
            {
                "id": skill.id,
                "name": skill.name,
                "salience": skill.salience,
                "prior": skill.prior,
                "precision": skill.precision,
                "applicable": self._is_applicable(skill),
                "preconditions": list(skill.preconditions),
                "missing_preconditions": self._missing_preconditions(skill),
                "triggers": [
                    self._trigger_diagnostic(trigger, event)
                    for trigger in skill.triggers
                ],
                "event": event_context,
            }
            for skill in ranked
        ]

    def save_snapshot(self) -> None:
        """Persist the current snapshot to the workspace database."""
        self._workspace.save_body_snapshot(
            json.dumps(self.snapshot(), ensure_ascii=False)
        )
