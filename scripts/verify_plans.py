"""
Independent re-simulation of the stored plans (scripts/verify_plans.py).

    python scripts/verify_plans.py                 # a sample of 200 plans
    python scripts/verify_plans.py --sample 0      # every stored plan
    python scripts/verify_plans.py --results other_dir
    python scripts/verify_plans.py --sample 0 --shards 4 --shard 0   # in parts

Every row of an experiment stores its plan as a JSON file. This script loads
those plans, rebuilds the instance and the hold-out sample from scratch, runs
the simulator again and compares the result with the numbers in the CSV.

It checks the recorded results in a way that does not depend on the solver:
if a row and its plan disagree, the numbers in the thesis would not be
reproducible from the stored plans. It needs no solver and takes about two
minutes for 700 plans.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd

from evaluation.due_dates import common_due_dates
from evaluation.holdout import HOLDOUT_SIZE, holdout_scenarios
from saa_scheduling.basics.instances import build_instance
from saa_scheduling.distributions.duration_profiles import build_case_models
from saa_scheduling.execution.plan import Plan
from saa_scheduling.execution.simulator import evaluate

#: relative tolerance of the comparison; the numbers should be identical
TOLERANCE = 1e-6


def capacities_of(label: str) -> dict[str, int]:
    """'Junior2-Senior1' -> {'Junior': 2, 'Senior': 1}."""
    return {part[:-1]: int(part[-1]) for part in label.split("-")}


class Holdout:
    """Rebuilds and caches the hold-out sample of an instance."""

    def __init__(self, size: int = HOLDOUT_SIZE):
        self._size = size
        self._cache: dict[tuple, tuple] = {}

    def of(self, process: str, cases: int, capacities: dict[str, int], level: str):
        key = (process, cases, tuple(sorted(capacities.items())), level)
        if key not in self._cache:
            instance = build_instance(process, cases, capacities)
            self._cache = {
                key: (instance, holdout_scenarios(process, cases, capacities, level, self._size))
            }
        return self._cache[key]


def rows_with_plans(results: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(results, "E*.csv"))):
        frame = pd.read_csv(path)
        if "plan_file" in frame:
            rows.extend(frame[frame.plan_file.notna()].to_dict("records"))
    return rows


def check(row: dict, holdout: Holdout) -> list[str]:
    """Re-evaluates one row; returns the names of the metrics that differ."""
    if not os.path.exists(row["plan_file"]):
        return ["plan file missing"]

    cases = int(row["cases"])
    capacities = capacities_of(row["capacities"])
    instance, scenarios = holdout.of(row["process"], cases, capacities, row["level"])
    stored = json.load(open(row["plan_file"]))
    plan = Plan.create(
        modes=stored["modes"], order=stored["order"], planned_starts=stored["planned_starts"]
    )

    due_dates = None
    if row.get("objective") == "sla":
        models = build_case_models(row["level"], cases)
        due_dates = common_due_dates(instance, models, float(row["sla_factor"]))

    realised = evaluate(instance, plan, scenarios, 0.0, due_dates)
    expected = [
        ("makespan", realised.mean_makespan, row["oos_makespan"]),
        ("on-time share", realised.on_time_share_at[0], row["oos_on_time_share"]),
        ("visibility", realised.visibility, row["oos_visibility"]),
    ]
    if due_dates is not None:
        expected += [
            ("tardiness", realised.mean_total_tardiness, row["oos_total_tardiness"]),
            ("violation rate", realised.sla_violation_rate, row["oos_sla_violation"]),
        ]
    return [
        f"{name}: {again:.6f} instead of {recorded:.6f}"
        for name, again, recorded in expected
        if not pd.isna(recorded) and abs(again - recorded) > TOLERANCE * max(1.0, abs(recorded))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--sample", type=int, default=200,
                        help="how many plans to check, 0 for all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shard", type=int, default=0,
                        help="check only every <shards>-th plan, starting at <shard>")
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)

    rows = rows_with_plans(args.results)
    if args.sample and args.sample < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)

    if args.shards > 1:
        rows = rows[args.shard :: args.shards]

    holdout = Holdout()
    failures = 0
    for row in rows:
        problems = check(row, holdout)
        for problem in problems:
            print(f"MISMATCH {row['plan_file']}: {problem}")
        failures += bool(problems)

    print(f"re-simulated {len(rows)} plans, {failures} with differences")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
