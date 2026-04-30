from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PlannerTrace:
    episode_id: str
    step: int
    team: str
    planner_state: dict[str, Any]
    macro_action: str
    reward_delta: float
    win: bool | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
