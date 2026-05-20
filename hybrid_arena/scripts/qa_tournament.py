"""Run QA tournament smoke evaluation."""

from __future__ import annotations

import argparse
import json

from hybrid_arena.qa.balance_report import write_reports
from hybrid_arena.qa.scenario_matrix import checkpoint_scenarios
from hybrid_arena.qa.tournament import run_tournament


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/qa/smoke")
    parser.add_argument(
        "--policy-source",
        choices=["rule_based", "checkpoint"],
        default="rule_based",
    )
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    scenarios = None
    if args.policy_source == "checkpoint":
        if not args.checkpoint:
            parser.error("--policy-source checkpoint requires --checkpoint")
        scenarios = checkpoint_scenarios(args.checkpoint)
    result = run_tournament(episodes=args.episodes, seed=args.seed, scenarios=scenarios)
    paths = write_reports(result, args.output)
    print(json.dumps({"result": result, "paths": paths}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
