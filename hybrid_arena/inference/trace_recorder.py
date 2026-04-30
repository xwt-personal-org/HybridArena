import json
from pathlib import Path

from hybrid_arena.inference.planner_trace import PlannerTrace


class PlannerTraceRecorder:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []

    def add(self, trace: PlannerTrace) -> None:
        self._buffer.append(trace.to_dict())

    def flush(self) -> None:
        if not self._buffer:
            return
        with self.output_path.open("w", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        self._buffer.clear()
