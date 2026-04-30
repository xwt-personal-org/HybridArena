"""Run a MiniMOBA game controlled by a macro-action planner."""

from __future__ import annotations

import argparse

from hybrid_arena.inference.adapter import MacroActionAdapter
from hybrid_arena.inference.llm_planner import DummyLLMClient, LLMPlanner
from hybrid_arena.inference.planner_state import summarize_game_state
from hybrid_arena.inference.rule_planner import RulePlanner
from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.env import parallel_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planner-controlled MiniMOBA demo")
    parser.add_argument("--planner", choices=["rule", "llm_dummy"], default="rule")
    parser.add_argument("--team", choices=["red", "blue"], default="red")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render-mode", choices=["rgb_array", "human", "none"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = parallel_env(map_size=16, team_size=2, max_steps=args.max_steps, render_mode=render_mode)
    obs, _ = env.reset(seed=42)

    planner = RulePlanner() if args.planner == "rule" else LLMPlanner(DummyLLMClient())
    random_agent = RandomAgent()
    macro_action = "group_mid"

    for step in range(args.max_steps):
        if step % 10 == 0:
            state = summarize_game_state(env.game_state, args.team)
            macro_action = planner.plan(state)
        adapter = MacroActionAdapter(macro_action)

        actions = {}
        for agent in env.agents:
            team = "red" if agent.startswith("red") else "blue"
            if team == args.team:
                actions[agent] = adapter.act(obs[agent])
            else:
                actions[agent] = random_agent.act(obs[agent])

        obs, _, terminations, truncations, _ = env.step(actions)
        if render_mode:
            env.render()
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()


if __name__ == "__main__":
    main()
