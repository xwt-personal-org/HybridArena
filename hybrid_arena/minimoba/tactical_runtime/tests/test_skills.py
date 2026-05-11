"""Tests for tactical skills and helper functions."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_arena.minimoba.tactical_runtime.body_schema import GameBodySchema
from hybrid_arena.minimoba.tactical_runtime.dispatcher import TacticalDispatcher
from hybrid_arena.minimoba.tactical_runtime.skills import (
    create_tactical_skills,
    direction_toward,
    nearest_tagged_region,
)
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


class TestDirectionToward:
    """Tests for direction_toward helper."""

    def test_stay_in_place(self):
        assert direction_toward((5, 5), (5, 5)) == 0

    def test_move_up(self):
        # target is above (y-1): delta (0, -1) → index 1
        assert direction_toward((5, 3), (5, 5)) == 1

    def test_move_down(self):
        # target is below (y+1): delta (0, 1) → index 5
        assert direction_toward((5, 7), (5, 5)) == 5

    def test_move_right(self):
        # target is right (x+1): delta (1, 0) → index 3
        assert direction_toward((7, 5), (5, 5)) == 3

    def test_move_left(self):
        # target is left (x-1): delta (-1, 0) → index 7
        assert direction_toward((3, 5), (5, 5)) == 7

    def test_move_northeast(self):
        # delta (1, -1) → index 2
        assert direction_toward((7, 3), (5, 5)) == 2

    def test_move_southeast(self):
        # delta (1, 1) → index 4
        assert direction_toward((7, 7), (5, 5)) == 4

    def test_move_southwest(self):
        # delta (-1, 1) → index 6
        assert direction_toward((3, 7), (5, 5)) == 6

    def test_move_northwest(self):
        # delta (-1, -1) → index 8
        assert direction_toward((3, 3), (5, 5)) == 8

    def test_far_target_uses_direction(self):
        # Far away target should still give correct direction
        assert direction_toward((100, 0), (5, 5)) == 2  # NE


class TestNearestTaggedRegion:
    """Tests for nearest_tagged_region helper."""

    def test_returns_none_when_empty(self):
        ws = BattlefieldWorkspace(map_size=32)
        assert nearest_tagged_region(ws, (5, 5), "resource_soon") is None

    def test_returns_nearest(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(10, 10), tags={"resource_soon"}))
        ws.add_annotation(BattlefieldAnnotation(position=(6, 6), tags={"resource_soon"}))
        ws.add_annotation(BattlefieldAnnotation(position=(20, 20), tags={"resource_soon"}))

        result = nearest_tagged_region(ws, (5, 5), "resource_soon")
        assert result == (6, 6)

    def test_ignores_other_tags(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(6, 6), tags={"danger"}))

        assert nearest_tagged_region(ws, (5, 5), "resource_soon") is None

    def test_exact_match(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(5, 5), tags={"objective"}))
        assert nearest_tagged_region(ws, (5, 5), "objective") == (5, 5)


class TestCreateTacticalSkills:
    """Tests for create_tactical_skills factory."""

    def test_returns_four_skills(self):
        ws = BattlefieldWorkspace(map_size=32)
        skills = create_tactical_skills(ws)
        assert len(skills) == 4

    def test_skill_ids(self):
        ws = BattlefieldWorkspace(map_size=32)
        skills = create_tactical_skills(ws)
        ids = {s.id for s in skills}
        assert ids == {"retreat_when_low", "farm_resources", "control_vision", "push_objective"}

    def test_all_have_triggers(self):
        ws = BattlefieldWorkspace(map_size=32)
        for skill in create_tactical_skills(ws):
            assert len(skill.triggers) > 0

    def test_all_have_controller(self):
        ws = BattlefieldWorkspace(map_size=32)
        for skill in create_tactical_skills(ws):
            assert skill.controller != ""


class TestRetreatSkill:
    """Test the retreat_when_low skill end-to-end."""

    def test_retreat_fires_on_low_health(self):
        ws = BattlefieldWorkspace(map_size=32)
        skills = create_tactical_skills(ws)
        retreat = next(s for s in skills if s.id == "retreat_when_low")

        gs = _StubGameState(heroes={
            "red_0": _StubHero(hp=200.0, x=16, y=16),  # 20% HP
        })
        trigger = retreat.triggers[0]
        score = trigger.score(ws, gs, "red_0")
        assert score > 0

    def test_retreat_does_not_fire_on_high_health(self):
        ws = BattlefieldWorkspace(map_size=32)
        skills = create_tactical_skills(ws)
        retreat = next(s for s in skills if s.id == "retreat_when_low")

        gs = _StubGameState(heroes={
            "red_0": _StubHero(hp=800.0, x=16, y=16),  # 80% HP
        })
        trigger = retreat.triggers[0]
        score = trigger.score(ws, gs, "red_0")
        assert score == 0.0


class TestFarmResourcesSkill:
    """Test the farm_resources skill end-to-end."""

    def test_farm_fires_with_resource_annotation(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(20, 20), tags={"resource_soon"}))
        skills = create_tactical_skills(ws)

        gs = _StubGameState(heroes={
            "red_0": _StubHero(hp=800.0, x=16, y=16),
        })
        event = GameEvent(kind="tick", agent_id="red_0")
        body = GameBodySchema(skills=skills, workspace=ws)
        dispatcher = TacticalDispatcher(body=body, workspace=ws)

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.success is True
        assert result.skill_id == "farm_resources"


class TestControlVisionSkill:
    """Test the control_vision skill end-to-end."""

    def test_vision_fires_with_dangerous_annotation(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(10, 10), tags={"dangerous"}))
        skills = create_tactical_skills(ws)

        gs = _StubGameState(heroes={
            "red_0": _StubHero(hp=800.0, x=16, y=16),
        })
        event = GameEvent(kind="tick", agent_id="red_0")
        body = GameBodySchema(skills=skills, workspace=ws)
        dispatcher = TacticalDispatcher(body=body, workspace=ws)

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.success is True
        assert result.skill_id == "control_vision"


class TestPushObjectiveSkill:
    """Test the push_objective skill end-to-end."""

    def test_push_fires_with_objective_annotation(self):
        ws = BattlefieldWorkspace(map_size=32)
        ws.add_annotation(BattlefieldAnnotation(position=(25, 25), tags={"objective"}))
        skills = create_tactical_skills(ws)

        gs = _StubGameState(heroes={
            "red_0": _StubHero(hp=800.0, x=16, y=16),
        })
        event = GameEvent(kind="tick", agent_id="red_0")
        body = GameBodySchema(skills=skills, workspace=ws)
        dispatcher = TacticalDispatcher(body=body, workspace=ws)

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.success is True
        assert result.skill_id in ("push_objective", "farm_resources")

    def test_push_fires_with_team_advantage(self):
        ws = BattlefieldWorkspace(map_size=32)
        skills = create_tactical_skills(ws)

        # Red team has advantage
        gs = _StubGameState(heroes={
            "red_0": _StubHero(hp=800.0, x=16, y=16),
        }, red_kills=5, blue_kills=1)
        event = GameEvent(kind="tick", agent_id="red_0")
        body = GameBodySchema(skills=skills, workspace=ws)
        dispatcher = TacticalDispatcher(body=body, workspace=ws)

        result = dispatcher.dispatch(event, game_state=gs, agent_id="red_0")
        assert result.success is True
        assert result.skill_id == "push_objective"
