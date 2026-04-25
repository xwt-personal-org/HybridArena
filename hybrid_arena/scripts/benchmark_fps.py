"""Benchmark environment performance (FPS)."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from hybrid_arena.minimoba.agents.random_agent import RandomAgent
from hybrid_arena.minimoba.env import parallel_env


def main():
    env = parallel_env(map_size=32, team_size=4, max_steps=5000)

    print("Warming up...")
    obs, _ = env.reset(seed=42)

    agent = RandomAgent()
    total_steps = 1000

    print(f"Running {total_steps} steps...")
    start = time.perf_counter()

    for _ in range(total_steps):
        actions = {a: agent.act(obs[a]) for a in env.agents}
        obs, rewards, terms, truncs, _ = env.step(actions)
        if not env.agents:
            obs, _ = env.reset()

    elapsed = time.perf_counter() - start
    fps = total_steps / elapsed
    agent_steps_per_sec = total_steps * len(env.possible_agents) / elapsed

    print("\n=== Benchmark Results ===")
    print(f"Total steps:    {total_steps}")
    print(f"Elapsed:        {elapsed:.2f}s")
    print(f"Env FPS:        {fps:.0f} steps/sec")
    print(f"Agent steps/s:  {agent_steps_per_sec:.0f}")
    print("Target:         > 500 FPS")

    if fps >= 500:
        print("PASSED")
    else:
        print(f"BELOW TARGET (missing by {500 - fps:.0f} FPS)")

    env.close()


if __name__ == "__main__":
    main()
