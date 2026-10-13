"""
Experiment entry point (scripts/run_experiments.py).

    python scripts/run_experiments.py smoke
    python scripts/run_experiments.py E1
    python scripts/run_experiments.py E2 --seconds 300 --workers 4
    python scripts/run_experiments.py E6 --seconds 120

Results are appended to results/<experiment>.csv. Re-running the same command
continues where the last run stopped: finished runs are skipped.

    caffeinate -i python scripts/run_experiments.py E2
"""

from __future__ import annotations

import argparse
import sys
import time

from evaluation.experiments import EXPERIMENTS, build_specs
from evaluation.holdout import HOLDOUT_SIZE
from evaluation.results import ResultsFile
from evaluation.runner import Runner
from saa_scheduling.solvers.base import SolverConfig


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS),
                        help="which experiment to run")
    parser.add_argument("--seconds", type=float, default=300.0, help="solve time limit per run")
    parser.add_argument("--workers", type=int, default=4, help="CP-SAT worker threads")
    parser.add_argument("--seed", type=int, default=0, help="solver seed")
    parser.add_argument("--holdout", type=int, default=HOLDOUT_SIZE, help="hold-out scenarios")
    parser.add_argument("--out", default=None, help="CSV path (default results/<experiment>.csv)")
    parser.add_argument("--limit", type=int, default=None, help="run at most this many specs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    specs = build_specs(args.experiment)
    if args.limit:
        specs = specs[: args.limit]

    path = args.out or f"results/{args.experiment}.csv"
    results = ResultsFile(path)
    config = SolverConfig(max_seconds=args.seconds, workers=args.workers, seed=args.seed)

    print(
        f"{args.experiment}: {len(specs)} runs, at most {args.seconds:.0f}s each, "
        f"{args.workers} workers, hold-out {args.holdout} scenarios",
        flush=True,
    )
    started = time.time()
    Runner(results, config, holdout_size=args.holdout).run_all(specs)
    minutes = (time.time() - started) / 60
    print(f"finished in {minutes:.1f} min, results in {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
