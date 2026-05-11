"""Game body schema: skill affordance tracking and prompt generation."""

from __future__ import annotations

from hybrid_arena.minimoba.tactical_runtime.schema import GameSkill
from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace, GameEvent


class GameBodySchema:
    """Tracks skill affordances based on workspace state and game events.

    The body schema maintains a list of available skills and updates their
    trigger scores as the game state changes. It provides ranked skill
    affordances for the dispatcher.
    """

    def __init__(self, skills: list[GameSkill], workspace: BattlefieldWorkspace) -> None:
        """Initialize the body schema.

        Args:
            skills: List of tactical skills to track.
            workspace: The battlefield workspace for trigger evaluation.
        """
        self.skills = list(skills)
        self.workspace = workspace
        self._last_scores: dict[str, float] = {}

    def update(self, event: GameEvent) -> None:
        """Update the body schema based on a new game event.

        Records the event in the workspace and re-evaluates all skill triggers.

        Args:
            event: The game event to process.
        """
        self.workspace.record_event(event)

    def current_affordances(
        self,
        game_state: object = None,
        agent_id: str = "",
        top_k: int = 8,
    ) -> list[GameSkill]:
        """Return the top-k skills ranked by trigger score.

        Evaluates all skill triggers against the current workspace state
        and returns the highest-scoring skills.

        Args:
            game_state: Optional game state for trigger evaluation.
            agent_id: The agent to evaluate skills for.
            top_k: Maximum number of skills to return.

        Returns:
            List of skills sorted by descending trigger score.
        """
        scored: list[tuple[float, GameSkill]] = []
        for skill in self.skills:
            max_score = 0.0
            for trigger in skill.triggers:
                score = trigger.score(self.workspace, game_state, agent_id)
                max_score = max(max_score, score)
            self._last_scores[skill.id] = max_score
            if max_score > 0:
                scored.append((max_score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]

    def to_prompt_summary(self) -> str:
        """Generate a text summary of current skill affordances for LLM context.

        Returns:
            Human-readable summary string of active skills and their scores.
        """
        lines = ["## Skill Affordances"]
        for skill in self.skills:
            score = self._last_scores.get(skill.id, 0.0)
            if score > 0:
                lines.append(f"- {skill.name} (id={skill.id}): score={score:.3f}")
        if len(lines) == 1:
            lines.append("- No active skills")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return a serializable snapshot of the current body schema state.

        Returns:
            Dictionary with skill IDs, scores, and metadata.
        """
        return {
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "score": self._last_scores.get(s.id, 0.0),
                    "no_go_traces": s.no_go_traces,
                    "prior": s.prior,
                }
                for s in self.skills
            ],
            "annotation_count": self.workspace.annotation_count,
            "event_count": self.workspace.event_count,
        }
