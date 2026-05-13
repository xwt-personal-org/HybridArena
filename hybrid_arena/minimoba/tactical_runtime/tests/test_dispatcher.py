from __future__ import annotations

from hybrid_arena.minimoba.tactical_runtime.dispatcher import TacticalDispatcher, TacticalSkill


def test_missing_controller_does_not_succeed() -> None:
    dispatcher = TacticalDispatcher()

    result = dispatcher.dispatch(
        [TacticalSkill(skill_id="bad-wiring", score=1.0, controller="missing_controller")],
        state={"hp": 1},
    )

    assert result.success is False
    assert result.action is None
    assert result.residual == 1.0
    assert result.message == "Unknown controller: missing_controller"
    assert dispatcher.trace[-1]["error"] == "unknown_controller"
    assert dispatcher.trace[-1]["controller"] == "missing_controller"


def test_invalid_controller_output_does_not_succeed() -> None:
    dispatcher = TacticalDispatcher(controllers={"bad_shape": lambda state: {"move": 0}})

    result = dispatcher.dispatch(
        [TacticalSkill(skill_id="bad-output", score=1.0, controller="bad_shape")],
        state={"hp": 1},
    )

    assert result.success is False
    assert result.action is None
    assert result.message == "Invalid controller action: bad_shape"
    assert dispatcher.trace[-1]["error"] == "invalid_controller_action"


def test_registered_controller_success_keeps_action() -> None:
    dispatcher = TacticalDispatcher(
        controllers={"retreat": lambda state: {"move": 1, "skill": 0, "target": 0}}
    )

    result = dispatcher.dispatch(
        [TacticalSkill(skill_id="retreat-now", score=1.0, controller="retreat")],
        state={"hp": 1},
    )

    assert result.success is True
    assert result.action == {"move": 1, "skill": 0, "target": 0}
    assert dispatcher.trace[-1]["success"] is True
