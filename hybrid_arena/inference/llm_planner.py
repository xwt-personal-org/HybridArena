"""LLM Planner: High-level tactical decision making.

Implements a simplified state machine (analyze -> decide -> reflect cycle)
that can be backed by either LangGraph or plain Python.

Output format (JSON):
    {
      "strategy": "团战/分推/发育/防守/抓人",
      "reasoning": "brief explanation",
      "assignments": {"tank": "task", "dps_1": "task", "dps_2": "task", "support": "task"},
      "target_positions": {"tank": [x,y], "dps_1": [x,y], ...}
    }
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from hybrid_arena.inference.macro_actions import validate_macro_action


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


class DummyLLMClient:
    def generate(self, prompt: str) -> str:
        return "group_mid"


class LLMPlanner:
    """High-level tactical planner.

    Supports three modes:
        - "mock": deterministic rule-based planner (no LLM dependency).
        - "local": local transformer model (Qwen 1.5B/3B).
        - "api": remote API (DeepSeek, etc.).
    """

    def __init__(
        self,
        client: BaseLLMClient | str | None = None,
        model_name: str | None = None,
        mode: str | None = None,
    ):
        if isinstance(client, str) and mode is None:
            mode = client
            client = None
        self.client = client
        self.mode = mode or ("client" if client is not None else "mock")
        self.model_name = model_name
        self._llm_fn: Callable | None = None

        if self.mode == "local":
            self._init_local()
        elif self.mode == "api":
            self._init_api()

    def _init_local(self) -> None:
        """Initialize local transformer model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = self.model_name or "Qwen/Qwen2.5-1.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._llm_fn = self._local_generate
        except Exception as e:
            print(f"[LLMPlanner] Local model init failed: {e}. Falling back to mock.")
            self.mode = "mock"

    def _init_api(self) -> None:
        """Initialize API client."""
        try:
            import openai

            self.api_client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
            )
            self.api_model = self.model_name or "deepseek-chat"
            self._llm_fn = self._api_generate
        except Exception as e:
            print(f"[LLMPlanner] API init failed: {e}. Falling back to mock.")
            self.mode = "mock"

    def _local_generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你是MOBA游戏战术AI。只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response

    def _api_generate(self, prompt: str) -> str:
        response = self.api_client.chat.completions.create(
            model=self.api_model,
            messages=[
                {"role": "system", "content": "你是MOBA游戏战术AI。只输出JSON。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _mock_generate(self, prompt: str) -> str:
        """Deterministic fallback when no LLM is available."""
        # Simple heuristic based on prompt keywords
        if "血量极低" in prompt or "阵亡" in prompt:
            strategy = "防守"
            reasoning = "队伍状态不佳，先防守发育"
        elif "可见敌方" in prompt and "4人" in prompt:
            strategy = "团战"
            reasoning = "敌方全员可见，可集合开团"
        elif "经济差" in prompt and "+" in prompt.split("经济差")[1].split("\n")[0]:
            strategy = "分推"
            reasoning = "经济领先，可分推扩大优势"
        else:
            strategy = "发育"
            reasoning = "局势不明，优先发育"

        return json.dumps(
            {
                "strategy": strategy,
                "reasoning": reasoning,
                "assignments": {
                    "tank": "保护后排" if strategy == "防守" else "先手开团",
                    "dps_1": "输出敌方核心",
                    "dps_2": "侧翼骚扰" if strategy == "分推" else "集火后排",
                    "support": "治疗保护",
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

    def _build_macro_prompt(self, planner_state) -> str:
        return (
            "Choose exactly one macro action from this set: "
            "group_mid, push_nearest_tower, retreat, farm_safe, protect_support, "
            "force_teamfight, split_push.\n"
            f"Planner state: {planner_state}\n"
            "Output only one macro action."
        )

    def plan(self, game_summary, history: list[str] | None = None):
        """Generate a tactical plan from game summary.

        Args:
            game_summary: Natural-language game state summary or PlannerState.
            history: Optional recent decision history.

        Returns:
            Parsed strategy dict.
        """
        if not isinstance(game_summary, str):
            prompt = self._build_macro_prompt(game_summary)
            if self.client is not None:
                response = self.client.generate(prompt)
            elif self.mode == "mock":
                response = DummyLLMClient().generate(prompt)
            else:
                response = self._llm_fn(prompt)
            return validate_macro_action(str(response).strip().split()[0])

        history_text = ""
        if history:
            history_text = "\n最近决策历史：\n" + "\n".join(history[-3:])

        prompt = f"""{game_summary}{history_text}

选择最优策略并为每个英雄分配任务。输出JSON格式：
{{"strategy":"团战/分推/发育/防守/抓人","reasoning":"原因","assignments":{{"tank":"任务","dps_1":"任务","dps_2":"任务","support":"任务"}},"target_positions":{{"tank":[x,y],"dps_1":[x,y],"dps_2":[x,y],"support":[x,y]}}}}"""

        if self.mode == "mock" or self._llm_fn is None:
            response = self._mock_generate(prompt)
        else:
            response = self._llm_fn(prompt)

        return self._parse_strategy(response)

    @staticmethod
    def _parse_strategy(text: str) -> dict:
        """Parse JSON from LLM response with fallback."""
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
            "reasoning": "JSON解析失败，执行默认策略",
            "assignments": {
                "tank": "巡逻",
                "dps_1": "清野",
                "dps_2": "清野",
                "support": "跟随坦克",
            },
            "target_positions": {
                "tank": [16, 16],
                "dps_1": [8, 8],
                "dps_2": [24, 24],
                "support": [16, 16],
            },
        }


class PlannerStateMachine:
    """Simplified state machine: analyze -> decide -> (optional) reflect."""

    def __init__(self, planner: LLMPlanner, reflect_interval: int = 5):
        self.planner = planner
        self.reflect_interval = reflect_interval
        self.state = TeamState()

    def step(self, game_summary: str) -> dict:
        """Run one planning cycle.

        Returns:
            Strategy dict with assignments and target positions.
        """
        self.state.game_summary = game_summary
        self.state.turn_count += 1

        # Every N steps, add a reflection step (simplified: just re-plan)
        if self.state.turn_count % self.reflect_interval == 0:
            self.state.reflection = f"第{self.state.turn_count}步重新评估局势"

        plan = self.planner.plan(game_summary, self.state.action_history)
        self.state.current_strategy = plan.get("strategy", "发育")
        self.state.hero_assignments = plan.get("assignments", {})
        self.state.action_history.append(
            f"Step {self.state.turn_count}: {self.state.current_strategy}"
        )
        return plan
