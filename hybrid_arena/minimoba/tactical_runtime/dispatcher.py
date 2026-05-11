"""Tactical dispatcher: bid-based skill selection with fallback."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_arena.minimoba.tactical_runtime.body_schema import GameBodySchema
from hybrid_arena.minimoba.tactical_runtime.schema import GameSkill
from hybrid_arena.minimoba.tactical_runtime.workspace import BattlefieldWorkspace, GameEvent


@dataclass
class TacticalDispatchResult:
    """Result of a tactical dispatch cycle.

    Contains the selected skill, the generated action, and diagnostic info.
    """

    skill_id: str | None = None
    action: dict | None = None  # {"move": int, "skill": int, "target": int}
    escalated: bool = False
    success: bool = False
    residual: float = 0.0
    message: str = ""


@dataclass
class _BidRecord:
    """Internal record of a skill bid for tracing."""

    skill_id: str
    trigger_score: float
    salience: float
    no_go_penalty: float
    bid: float


class TacticalDispatcher:
    """Bid-based tactical skill dispatcher.

    Evaluates skill triggers, computes bids, selects the highest positive bid,
    and generates an action. Falls back to a fallback planner if no skill wins.
    """

    def __init__(
        self,
        body: GameBodySchema,
        workspace: BattlefieldWorkspace,
        fallback_planner: object = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            body: The body schema for skill affordance tracking.
            workspace: The battlefield workspace.
            fallback_planner: Optional fallback planner (callable or object with dispatch).
        """
        self.body = body
        self.workspace = workspace
        self.fallback_planner = fallback_planner
        self._trace: list[dict] = []

    def dispatch(
        self,
        event: GameEvent,
        game_state: object = None,
        agent_id: str = "",
    ) -> TacticalDispatchResult:
        """Dispatch a tactical action in response to a game event.

        Evaluates all skill triggers, computes bids, selects the winner,
        and generates an action dict.

        Bid formula: trigger_score * skill.salience - tonic_inhibition - no_go_penalty
            - tonic_inhibition = 0.0
            - no_go_penalty = skill.no_go_traces * 0.1

        Args:
            event: The triggering game event.
            game_state: Optional game state for trigger evaluation.
            agent_id: The agent to dispatch for.

        Returns:
            TacticalDispatchResult with the selected action or escalation.
        """
        # Update body schema with the event
        self.body.update(event)

        # Evaluate all skills and compute bids
        bids: list[_BidRecord] = []
        for skill in self.body.skills:
            max_trigger_score = 0.0
            for trigger in skill.triggers:
                score = trigger.score(self.workspace, game_state, agent_id)
                max_trigger_score = max(max_trigger_score, score)

            if max_trigger_score <= 0:
                continue

            tonic_inhibition = 0.0
            no_go_penalty = skill.no_go_traces * 0.1
            bid = max_trigger_score * skill.salience - tonic_inhibition - no_go_penalty

            bids.append(_BidRecord(
                skill_id=skill.id,
                trigger_score=max_trigger_score,
                salience=skill.salience,
                no_go_penalty=no_go_penalty,
                bid=bid,
            ))

        # Sort by bid descending
        bids.sort(key=lambda b: b.bid, reverse=True)

        # Record trace
        self._trace.append({
            "event_kind": event.kind,
            "agent_id": agent_id,
            "bids": [
                {"skill_id": b.skill_id, "bid": round(b.bid, 4)}
                for b in bids
            ],
        })

        # Select winner: highest positive bid
        if bids and bids[0].bid > 0:
            winner = bids[0]
            skill = self._find_skill(winner.skill_id)
            if skill is not None:
                action = self._generate_action(skill, event, game_state, agent_id)
                return TacticalDispatchResult(
                    skill_id=winner.skill_id,
                    action=action,
                    escalated=False,
                    success=True,
                    residual=winner.bid,
                    message=f"Skill '{skill.name}' selected with bid {winner.bid:.3f}",
                )

        # Fallback
        if self.fallback_planner is not None:
            result = self._call_fallback(event, game_state, agent_id)
            if result is not None:
                return result

        # No skill matched, no fallback
        return TacticalDispatchResult(
            skill_id=None,
            action=None,
            escalated=True,
            success=False,
            residual=0.0,
            message="No skill matched and no fallback available",
        )

    def _find_skill(self, skill_id: str) -> GameSkill | None:
        """Find a skill by ID in the body schema."""
        for skill in self.body.skills:
            if skill.id == skill_id:
                return skill
        return None

    def _generate_action(
        self,
        skill: GameSkill,
        event: GameEvent,
        game_state: object,
        agent_id: str,
    ) -> dict:
        """Generate an action dict from a skill and context.

        Delegates to the skill's controller function if available.

        Returns:
            Action dict with keys: move, skill, target.
        """
        # Look up controller function from skill.controller name
        controller_fn = _CONTROLLER_REGISTRY.get(skill.controller)
        if controller_fn is not None:
            return controller_fn(skill, event, self.workspace, game_state, agent_id)
        # Default: stay in place, no skill, no target
        return {"move": 0, "skill": 0, "target": 0}

    def _call_fallback(
        self,
        event: GameEvent,
        game_state: object,
        agent_id: str,
    ) -> TacticalDispatchResult | None:
        """Call the fallback planner if available."""
        planner = self.fallback_planner
        if planner is None:
            return None
        if callable(planner):
            action = planner(event, game_state, agent_id)
            if action is not None:
                return TacticalDispatchResult(
                    skill_id=None,
                    action=action,
                    escalated=True,
                    success=True,
                    residual=0.0,
                    message="Fallback planner provided action",
                )
        return None

    @property
    def trace(self) -> list[dict]:
        """Return the in-memory dispatch trace."""
        return self._trace


# Controller function registry — populated by skills.py
_CONTROLLER_REGISTRY: dict[str, object] = {}


def register_controller(name: str, fn: object) -> None:
    """Register a controller function by name.

    Args:
        name: Controller name (matches GameSkill.controller field).
        fn: Controller function with signature (skill, event, workspace, game_state, agent_id) -> dict.
    """
    _CONTROLLER_REGISTRY[name] = fn
