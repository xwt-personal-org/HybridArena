"""Tests for multi-agent tactical dispatch conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_arena.minimoba.tactical_runtime.body_schema import GameBodySchema
from hybrid_arena.minimoba.tactical_runtime.skills import create_tactical_skills
from hybrid_arena.minimoba.tactical_runtime.team_dispatcher import TeamTacticalDispatcher
from hybrid_arena.minimoba.tactical_runtime.workspace import (
    BattlefieldAnnotation,
    BattlefieldWorkspace,
    GameEvent,
)


@dataclass
class _StubHeroConfig:
    max_hp: float = 1000.0


@dataclass
class _StubHero:
    hero_id: str
    team: str = "red"
    x: int = 0
    y: int = 0
    hp: float = 800.0
    alive: bool = True
    config: _StubHeroConfig | None = None

    def __post_init__(self):
        if self.config is None:
            self.config = _StubHeroConfig()


@dataclass
class _StubGameState:
    heroes: dict
    red_kills: int = 0
    blue_kills: int = 0
    red_towers: int = 2
    blue_towers: int = 2


class TestTeamTacticalDispatcher:
    """Tests for team-level dispatching."""

    def _make_dispatcher(self, workspace):
        skills = create_tactical_skills(workspace)
        body = GameBodySchema(skills=skills, workspace=workspace)
        return TeamTacticalDispatcher(body=body, workspace=workspace)

    def test_dispatch_team_returns_action_per_agent(self):
        workspace = BattlefieldWorkspace(map_size=32)
        workspace.add_annotation(BattlefieldAnnotation(
            position=(20, 20),
            tags={"resource_soon"},
            intensity=0.8,
        ))
        dispatcher = self._make_dispatcher(workspace)
        state = _StubGameState(heroes={
            "red_0": _StubHero("red_0", x=10, y=10),
            "red_1": _StubHero("red_1", x=12, y=12),
        })

        result = dispatcher.dispatch_team(
            events={
                "red_0": GameEvent(kind="tick", agent_id="red_0"),
                "red_1": GameEvent(kind="tick", agent_id="red_1"),
            },
            game_state=state,
            agent_ids=["red_0", "red_1"],
        )

        assert set(result.actions) == {"red_0", "red_1"}
        for action in result.actions.values():
            assert set(action) == {"move", "skill", "target"}

    def test_same_resource_conflict_keeps_closest_agent(self):
        workspace = BattlefieldWorkspace(map_size=32)
        workspace.add_annotation(BattlefieldAnnotation(
            position=(20, 20),
            tags={"resource_soon"},
            intensity=0.8,
        ))
        dispatcher = self._make_dispatcher(workspace)
        state = _StubGameState(heroes={
            "red_0": _StubHero("red_0", x=19, y=20),
            "red_1": _StubHero("red_1", x=5, y=5),
        })

        result = dispatcher.dispatch_team(
            events={
                "red_0": GameEvent(kind="tick", agent_id="red_0"),
                "red_1": GameEvent(kind="tick", agent_id="red_1"),
            },
            game_state=state,
            agent_ids=["red_0", "red_1"],
        )

        assert result.actions["red_0"] != {"move": 0, "skill": 0, "target": 0}
        assert result.actions["red_1"] == {"move": 0, "skill": 0, "target": 0}
        assert result.conflicts == [
            {
                "target": (20, 20),
                "kept_agent": "red_0",
                "rerouted_agents": ["red_1"],
            }
        ]

    def test_conflict_tie_breaks_by_agent_id(self):
        workspace = BattlefieldWorkspace(map_size=32)
        workspace.add_annotation(BattlefieldAnnotation(
            position=(20, 20),
            tags={"objective"},
            intensity=0.8,
        ))
        dispatcher = self._make_dispatcher(workspace)
        state = _StubGameState(heroes={
            "red_1": _StubHero("red_1", x=19, y=20),
            "red_0": _StubHero("red_0", x=20, y=19),
        })

        result = dispatcher.dispatch_team(
            events={
                "red_1": GameEvent(kind="tick", agent_id="red_1"),
                "red_0": GameEvent(kind="tick", agent_id="red_0"),
            },
            game_state=state,
            agent_ids=["red_1", "red_0"],
        )

        assert result.actions["red_0"] != {"move": 0, "skill": 0, "target": 0}
        assert result.actions["red_1"] == {"move": 0, "skill": 0, "target": 0}
        assert result.conflicts[0]["kept_agent"] == "red_0"

