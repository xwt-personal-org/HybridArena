"""Tests for the tactical dispatcher."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_arena.minimoba.tactical_runtime.body_schema import GameBodySchema
from hybrid_arena.minimoba.tactical_runtime.dispatcher import (
    TacticalDispatcher,
    TacticalDispatchResult,
)
from hybrid_arena.minimoba.tactical_runtime.skills import create_tactical_skills
from hybrid_arena.minimoba.tactical_runtime.workspace import (
    BattlefieldAnnotation,
    BattlefieldWorkspace,
    GameEvent,
)

# --- Minimal game state stubs for testing ---


@dataclass
class _StubHeroConfig:
    max_hp: float = 1000.0


@dataclass
class _StubHero:
    hero_id: str = "red_0"
    team: str = "red"
    x: int = 16
    y: int = 16
    hp: float = 500.0
    alive: bool = True
    config: _StubHeroConfig = None

    def __post_init__(self):
        if self.config is None:
            self.config = _StubHeroConfig()


@dataclass
class _StubGameState:
    heroes: dict = None
    red_kills: int = 0
    blue_kills: int = 0
    red_towers: int = 2
    blue_towers: int = 2

    def __post_init__(self):
        if self.heroes is None:
            self.heroes = {}


class TestTacticalDispatchResult:
    """Tests for TacticalDispatchResult dataclass."""

    def test_defaults(self):
        r = TacticalDispatchResult()
        assert r.skill_id is None
        assert r.action is None
        assert r.escalated is False
        assert r.success is False
        assert r.residual == 0.0
        assert r.message == ""


class TestTacticalDispatcher:
    """Tests for TacticalDispatcher."""

    def _make_dispatcher(self, skills=None, fallback=None):
        ws = BattlefieldWorkspace(map_size=32)
        if skills is None:
            skills = create_tactical_skills(ws)
        body = GameBodySchema(skills=skills, workspace=ws)
        return TacticalDispatcher(body=body, workspace=ws, fallback_planner=fallback), ws

    def test_low_health_selects_retreat(self):
        dispatcher, ws = self._make_dispatcher()
        hero = _StubHero(hp=200.0)  # 200/1000 = 0.2, below 0.25
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0", position=(16, 16))

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.success is True
        assert result.skill_id == "retreat_when_low"
        assert result.action is not None
        assert set(result.action.keys()) == {"move", "skill", "target"}

    def test_retreat_action_moves_toward_base(self):
        dispatcher, ws = self._make_dispatcher()
        hero = _StubHero(hp=200.0, x=16, y=16)
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0", position=(16, 16))

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        # Should move toward (0, 0) → direction 8 (NW: -1, -1)
        assert result.action["move"] == 8

    def test_nearby_enemy_selects_relevant_skill(self):
        dispatcher, ws = self._make_dispatcher()
        # Add danger annotation near the hero
        ws.add_annotation(BattlefieldAnnotation(
            position=(17, 17), tags={"dangerous"}, intensity=0.9
        ))
        hero = _StubHero(hp=800.0, x=16, y=16)
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0", position=(16, 16))

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.success is True
        assert result.skill_id == "control_vision"

    def test_no_match_escalates(self):
        dispatcher, ws = self._make_dispatcher()
        hero = _StubHero(hp=800.0)  # healthy, no annotations
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0")

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.escalated is True
        assert result.success is False

    def test_fallback_called_when_no_match(self):
        def fake_planner(event, game_state, agent_id):
            return {"move": 0, "skill": 1, "target": 2}

        dispatcher, ws = self._make_dispatcher(fallback=fake_planner)
        hero = _StubHero(hp=800.0)
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0")

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.escalated is True
        assert result.success is True
        assert result.action == {"move": 0, "skill": 1, "target": 2}

    def test_action_has_correct_keys(self):
        dispatcher, ws = self._make_dispatcher()
        hero = _StubHero(hp=200.0, x=10, y=10)
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0", position=(10, 10))

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert isinstance(result.action, dict)
        assert "move" in result.action
        assert "skill" in result.action
        assert "target" in result.action
        assert isinstance(result.action["move"], int)
        assert isinstance(result.action["skill"], int)
        assert isinstance(result.action["target"], int)

    def test_trace_recorded(self):
        dispatcher, ws = self._make_dispatcher()
        hero = _StubHero(hp=200.0)
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0")

        dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert len(dispatcher.trace) == 1
        assert dispatcher.trace[0]["event_kind"] == "tick"
        assert dispatcher.trace[0]["agent_id"] == "red_0"

    def test_annotation_query_trigger_with_objective(self):
        dispatcher, ws = self._make_dispatcher()
        ws.add_annotation(BattlefieldAnnotation(
            position=(20, 20), tags={"objective"}, intensity=0.8
        ))
        hero = _StubHero(hp=800.0, x=16, y=16)
        gs = _StubGameState(heroes={"red_0": hero})
        event = GameEvent(kind="tick", agent_id="red_0")

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.success is True
        assert result.skill_id in ("push_objective", "farm_resources")
