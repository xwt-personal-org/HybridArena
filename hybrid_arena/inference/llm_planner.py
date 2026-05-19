"""LLM planner interfaces with strict macro-decision validation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from hybrid_arena.inference.macro_actions import canonical_macro_action, validate_macro_action
from hybrid_arena.inference.prompt_templates import build_macro_prompt


@dataclass
class TeamState:
    """Internal state for the planner cycle."""

    game_summary: str = ""
    action_history: list[str] = field(default_factory=list)
    current_strategy: str = "balanced"
    hero_assignments: dict = field(default_factory=dict)
    reflection: str = ""
    turn_count: int = 0


class BaseLLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class LLMPlannerProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class StubLLMProvider:
    def __init__(self, macro_action: str = "GROUP_MID"):
        self.macro_action = macro_action

    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "macro_action": self.macro_action,
                "reasoning": "deterministic stub provider",
                "reward_bias": {},
                "action_mask_bias": {},
            },
            sort_keys=True,
        )


class DummyLLMClient(StubLLMProvider):
    def __init__(self):
        super().__init__("group_mid")


def validate_llm_decision(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"macro_action", "reasoning", "reward_bias", "action_mask_bias"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Missing LLM decision fields: {sorted(missing)}")
    macro_action = validate_macro_action(str(payload["macro_action"]))
    if not isinstance(payload["reasoning"], str):
        raise ValueError("reasoning must be a string")
    for field_name in ("reward_bias", "action_mask_bias"):
        if not isinstance(payload[field_name], dict):
            raise ValueError(f"{field_name} must be a dict")
        for key, value in payload[field_name].items():
            if not isinstance(key, str) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} entries must be numeric")
    return {
        "macro_action": macro_action,
        "canonical_macro_action": canonical_macro_action(macro_action),
        "reasoning": payload["reasoning"],
        "reward_bias": dict(payload["reward_bias"]),
        "action_mask_bias": dict(payload["action_mask_bias"]),
    }


class LLMPlanner:
    """High-level planner.

    PlannerState inputs return a macro action string. Legacy natural-language
    summaries still return the older strategy dict used by existing demos.
    """

    def __init__(
        self,
        client: BaseLLMClient | str | None = None,
        model_name: str | None = None,
        mode: str | None = None,
        provider: LLMPlannerProvider | None = None,
    ):
        if isinstance(client, str) and mode is None:
            mode = client
            client = None
        self.client = client
        self.provider = provider
        self.mode = mode or ("client" if client is not None else "mock")
        self.model_name = model_name
        self._llm_fn: Callable[[str], str] | None = None

    def _generate_macro_json(self, prompt: str) -> str:
        if self.provider is not None:
            return self.provider.generate(prompt)
        if self.client is not None:
            return self.client.generate(prompt)
        return StubLLMProvider().generate(prompt)

    def _build_macro_prompt(self, planner_state) -> str:
        return build_macro_prompt(planner_state)

    def plan(self, game_summary, history: list[str] | None = None):
        if not isinstance(game_summary, str):
            prompt = self._build_macro_prompt(game_summary)
            decision = self._parse_macro_decision(self._generate_macro_json(prompt))
            return decision["macro_action"]

        prompt = self._build_legacy_prompt(game_summary, history)
        response = self._mock_generate(prompt) if self.mode == "mock" else self._mock_generate(prompt)
        return self._parse_strategy(response)

    @staticmethod
    def _parse_macro_decision(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM macro output must be valid JSON") from exc
        return validate_llm_decision(payload)

    def _build_legacy_prompt(self, game_summary: str, history: list[str] | None = None) -> str:
        history_text = ""
        if history:
            history_text = "\nRecent decisions:\n" + "\n".join(history[-3:])
        return f"{game_summary}{history_text}"

    def _mock_generate(self, prompt: str) -> str:
        if "+" in prompt or "visible" in prompt:
            strategy = "团战"
            reasoning = "mock planner sees an advantage"
        else:
            strategy = "发育"
            reasoning = "mock planner default"
        return json.dumps(
            {
                "strategy": strategy,
                "reasoning": reasoning,
                "assignments": {
                    "tank": "frontline",
                    "dps_1": "damage",
                    "dps_2": "side pressure",
                    "support": "protect",
                },
                "target_positions": {
                    "tank": [16, 16],
                    "dps_1": [12, 12],
                    "dps_2": [20, 20],
                    "support": [16, 16],
                },
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _parse_strategy(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {
            "strategy": "发育",
            "reasoning": "JSON parse failed; using default strategy.",
            "assignments": {
                "tank": "patrol",
                "dps_1": "farm",
                "dps_2": "farm",
                "support": "follow tank",
            },
            "target_positions": {
                "tank": [16, 16],
                "dps_1": [8, 8],
                "dps_2": [24, 24],
                "support": [16, 16],
            },
        }


class PlannerStateMachine:
    """Simplified state machine: analyze -> decide -> optional reflect."""

    def __init__(self, planner: LLMPlanner, reflect_interval: int = 5):
        self.planner = planner
        self.reflect_interval = reflect_interval
        self.state = TeamState()

    def step(self, game_summary: str) -> dict:
        self.state.game_summary = game_summary
        self.state.turn_count += 1
        if self.state.turn_count % self.reflect_interval == 0:
            self.state.reflection = f"Step {self.state.turn_count}: refresh tactical assessment"
        plan = self.planner.plan(game_summary, self.state.action_history)
        self.state.current_strategy = plan.get("strategy", "发育")
        self.state.hero_assignments = plan.get("assignments", {})
        self.state.action_history.append(
            f"Step {self.state.turn_count}: {self.state.current_strategy}"
        )
        return plan
