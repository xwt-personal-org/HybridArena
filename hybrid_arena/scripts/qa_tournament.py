"""Run QA tournament smoke evaluation."""

from __future__ import annotations

import argparse
import json

from hybrid_arena.qa.balance_report import write_reports
from hybrid_arena.qa.tournament import run_tournament


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/qa/smoke")
    args = parser.parse_args()
    result = run_tournament(episodes=args.episodes, seed=args.seed)
    paths = write_reports(result, args.output)
    print(json.dumps({"result": result, "paths": paths}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
