"""Run a local two-process-style actor/learner smoke test."""

from __future__ import annotations

import argparse
import json

from hybrid_arena.distributed.actor_worker import LocalActorWorker
from hybrid_arena.distributed.learner_worker import LocalLearnerWorker
from hybrid_arena.distributed.policy_sync import PolicyStore
from hybrid_arena.distributed.replay_queue import BoundedReplayQueue


def run_smoke(actors: int, steps: int, seed: int) -> dict:
    policy_store = PolicyStore()
    queue = BoundedReplayQueue(max_chunks=max(actors, 1), drop_oldest=False)
    actor_workers = [
        LocalActorWorker(f"actor-{index}", policy_store, seed=seed + index)
        for index in range(actors)
    ]
    actor_fps_values = []
    per_actor_steps = max(1, steps // max(actors, 1))
    for worker in actor_workers:
        chunk = worker.collect(per_actor_steps)
        pushed = queue.push(chunk)
        if not pushed:
            queue.dropped_chunks += 1
        actor_fps_values.append(worker.last_fps)

    learner = LocalLearnerWorker(policy_store)
    metrics = learner.consume(queue, actor_fps=sum(actor_fps_values))
    return metrics.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.actors, args.steps, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
