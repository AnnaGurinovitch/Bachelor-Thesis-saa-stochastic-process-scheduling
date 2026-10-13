"""
Experiment definitions (src/evaluation/experiments.py).

  E1   convergence and value of SAA on an easy instance
  E1b  the same on a hard instance near the tractability boundary
  E2   tractability of the two formulations on growing instances
  E3   price of visibility: the weight lambda versus published quantiles
  E3b  E3 again, with the tolerant visibility metrics and stored plans
  E4   main evaluation: the whole complexity grid, three seeds, borders of
       visibility, value of SAA and tractability in one systematic sweep
  E5   time-limit study: does the tractability boundary move with more time?
  E6   value of the stochastic solution: mixed-risk cases and case deadlines,
       the settings in which a mean-based planner is blind by construction

Every experiment is a list of RunSpec records. A spec is a complete
description of one row of results. Specs that differ only in how the start
times are published share one solve; the runner caches it.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.results import PUBLISH_NONE, PUBLISH_OPTIMISED, PUBLISH_OWN
from saa_scheduling.basics.instances import JUNIOR, SENIOR

METHOD_CP = "saa_cp"
METHOD_MSEQCT = "saa_mseqct"
METHOD_DET = "det"
METHOD_ROB = "rob"

TIGHT = {JUNIOR: 1, SENIOR: 1}
MEDIUM = {JUNIOR: 2, SENIOR: 1}
WIDE = {JUNIOR: 2, SENIOR: 2}

SCENARIO_GRID = (5, 10, 20, 50, 100)
LEVELS = ("D1", "D2", "D3")
LAMBDA_GRID = (0.0, 0.05, 0.2, 1.0, 5.0)
QUANTILE_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)


def quantile_label(probability: float) -> str:
    return f"q{round(probability * 100)}"


def quantile_of(publish: str) -> float | None:
    """The probability behind a publication label such as 'q90', else None."""
    if publish.startswith("q") and publish[1:].isdigit():
        return int(publish[1:]) / 100
    return None


PUBLISH_QUANTILES = tuple(quantile_label(q) for q in QUANTILE_GRID)


@dataclass(frozen=True)
class RunSpec:
    """One result row: instance, level, method, its parameters and publication."""

    experiment: str
    process: str
    cases: int
    capacities: dict
    level: str
    method: str
    n_scenarios: int = 1
    visibility_weight: float = 0.0
    seed: int = 0
    publish: str = PUBLISH_NONE
    #: overrides the command line time limit for this spec (E5)
    max_seconds: float | None = None
    #: "makespan", or "sla" for case tardiness against a common due date
    objective: str = "makespan"
    #: due date as a multiple of the mean-based reference makespan ("sla" only)
    sla_factor: float | None = None

    @property
    def sla_label(self) -> str:
        return "" if self.sla_factor is None else f"{self.sla_factor:g}"

    @property
    def time_limit_label(self) -> str:
        return "" if self.max_seconds is None else str(int(self.max_seconds))

    @property
    def capacity_label(self) -> str:
        return "-".join(f"{role}{self.capacities[role]}" for role in sorted(self.capacities))

    @property
    def solve_key(self) -> tuple:
        """Specs with the same solve key share one solver run."""
        return (
            self.process, self.cases, self.capacity_label, self.level,
            self.method, self.n_scenarios, self.visibility_weight, self.seed,
            self.max_seconds, self.objective, self.sla_factor,
        )

    @property
    def label(self) -> str:
        return (
            f"{self.experiment}/{self.process}/{self.cases}cases/{self.capacity_label}/"
            f"{self.level}/{self.method}/N{self.n_scenarios}/lam{self.visibility_weight}/"
            f"seed{self.seed}/{self.publish}"
            + (f"/limit{self.time_limit_label}s" if self.max_seconds is not None else "")
            + (f"/sla{self.sla_label}" if self.objective == "sla" else "")
        )


def baseline_specs(experiment, process, cases, capacities, level) -> list[RunSpec]:
    """DET and ROB, each with and without their own start times."""
    return [
        RunSpec(experiment, process, cases, capacities, level, method, publish=publish)
        for method in (METHOD_DET, METHOD_ROB)
        for publish in (PUBLISH_OWN, PUBLISH_NONE)
    ]


def saa_specs(experiment, process, cases, capacities, level, n, seed, publishes, method=METHOD_CP):
    return [
        RunSpec(experiment, process, cases, capacities, level, method, n, 0.0, seed, publish)
        for publish in publishes
    ]


def e1_convergence(seeds: int = 5) -> list[RunSpec]:
    """P1, two cases, medium capacity: vary N and the variability level."""
    specs: list[RunSpec] = []
    for level in LEVELS:
        specs += baseline_specs("E1", "P1", 2, MEDIUM, level)
        for n in SCENARIO_GRID:
            for seed in range(seeds):
                specs += saa_specs("E1", "P1", 2, MEDIUM, level, n, seed, (PUBLISH_NONE,))
    return specs


def e1b_convergence_hard(seeds: int = 3) -> list[RunSpec]:
    """P1, three cases, tight capacity: near the tractability boundary."""
    specs: list[RunSpec] = []
    for level in LEVELS:
        specs += baseline_specs("E1b", "P1", 3, TIGHT, level)
        for n in SCENARIO_GRID:
            for seed in range(seeds):
                specs += saa_specs("E1b", "P1", 3, TIGHT, level, n, seed, (PUBLISH_NONE,))
    return specs


def e2_tractability(seeds: int = 3) -> list[RunSpec]:
    """Both formulations on growing instances; ordered small to large."""
    specs: list[RunSpec] = []
    for cases in (1, 2, 3, 4):
        for process in ("P1", "P2", "P3"):
            for capacities in (TIGHT, WIDE):
                for n in (10, 100):
                    for method in (METHOD_CP, METHOD_MSEQCT):
                        for seed in range(seeds):
                            specs += saa_specs(
                                "E2", process, cases, capacities, "D2", n, seed,
                                (PUBLISH_NONE,), method,
                            )
    return specs


def e3_visibility(seeds: int = 5) -> list[RunSpec]:
    """P1 and P3 under strong bimodality: the weight lambda versus quantiles."""
    specs: list[RunSpec] = []
    for process in ("P1", "P3"):
        specs += baseline_specs("E3", process, 2, MEDIUM, "D3")
        for seed in range(seeds):
            specs += saa_specs(
                "E3", process, 2, MEDIUM, "D3", 50, seed,
                (PUBLISH_NONE,) + PUBLISH_QUANTILES,
            )
        for weight in LAMBDA_GRID[1:]:
            for seed in range(seeds):
                specs.append(
                    RunSpec("E3", process, 2, MEDIUM, "D3", METHOD_CP, 50, weight, seed,
                            PUBLISH_OPTIMISED)
                )
    return specs


def e3b_visibility(seeds: int = 5) -> list[RunSpec]:
    """E3 recomputed under a new name, so every row has all metrics and a plan."""
    return [
        RunSpec(**{**spec.__dict__, "experiment": "E3b"}) for spec in e3_visibility(seeds)
    ]


def e4_main_evaluation(n_scenarios: int = 20, seeds: int = 3) -> list[RunSpec]:
    """The main evaluation: every process, case count, capacity and level.

    Per instance, both baselines (with and without start times) and, for every
    seed, one SAA solve evaluated with every publication. P1 is additionally
    run with five and six cases to find where it breaks down completely.
    Ordered from small to large, so a stopped run keeps the easy part complete;
    seed 0 of cases 1-4 is the original E4 and is skipped if already present.
    """
    grid = [
        (cases, process)
        for cases in (1, 2, 3, 4)
        for process in ("P1", "P2", "P3")
    ] + [(5, "P1"), (6, "P1")]

    specs: list[RunSpec] = []
    for cases, process in grid:
        for capacities in (TIGHT, MEDIUM, WIDE):
            for level in LEVELS:
                specs += baseline_specs("E4", process, cases, capacities, level)
                for seed in range(seeds):
                    specs += saa_specs(
                        "E4", process, cases, capacities, level, n_scenarios, seed,
                        (PUBLISH_NONE,) + PUBLISH_QUANTILES,
                    )
    return specs


TIME_LIMITS = (60.0, 300.0, 1200.0)


def e5_time_limits() -> list[RunSpec]:
    """Both formulations on the hard instances with three time limits."""
    specs: list[RunSpec] = []
    for process in ("P1", "P3"):
        for cases in (3, 4):
            for method in (METHOD_CP, METHOD_MSEQCT):
                for limit in TIME_LIMITS:
                    specs.append(
                        RunSpec("E5", process, cases, TIGHT, "D2", method, 20, 0.0, 0,
                                PUBLISH_NONE, limit)
                    )
    return specs


OBJECTIVES = (("makespan", None), ("sla", 0.85), ("sla", 1.0))


def e6_value_of_stochastic_solution(n_scenarios: int = 20, seeds: int = 3) -> list[RunSpec]:
    """Where the mean is blind: homogeneous risky cases (D3) versus mixed cases
    (MIX: every second case risky, all with the same means), each under the
    makespan objective and under a tight and a medium case deadline.

    The DET baseline sees identical cases in MIX by construction; the value of
    the stochastic solution is the difference between DET and SAA.
    """
    specs: list[RunSpec] = []
    for cases in (2, 3, 4):
        for process in ("P1", "P3"):
            for capacities in (TIGHT, MEDIUM):
                for level in ("D3", "MIX"):
                    for objective, factor in OBJECTIVES:
                        common = dict(objective=objective, sla_factor=factor)
                        for method in (METHOD_DET, METHOD_ROB):
                            specs.append(RunSpec("E6", process, cases, capacities, level,
                                                 method, publish=PUBLISH_NONE, **common))
                        for seed in range(seeds):
                            specs.append(RunSpec("E6", process, cases, capacities, level,
                                                 METHOD_CP, n_scenarios, 0.0, seed,
                                                 PUBLISH_NONE, **common))
    return specs


def smoke() -> list[RunSpec]:
    """A few minutes of runs that exercise every method and publication."""
    specs = baseline_specs("smoke", "P1", 2, MEDIUM, "D3")
    specs += saa_specs("smoke", "P1", 2, MEDIUM, "D3", 10, 0, (PUBLISH_NONE,) + PUBLISH_QUANTILES)
    specs += saa_specs("smoke", "P1", 2, MEDIUM, "D3", 10, 0, (PUBLISH_NONE,), METHOD_MSEQCT)
    specs.append(RunSpec("smoke", "P1", 2, MEDIUM, "D3", METHOD_CP, 5, 1.0, 0, PUBLISH_OPTIMISED))
    specs += saa_specs("smoke", "P3", 2, MEDIUM, "D2", 10, 0, (PUBLISH_NONE, "q90"))
    for method in (METHOD_DET, METHOD_CP):
        specs.append(RunSpec("smoke", "P1", 2, TIGHT, "MIX", method, 10 if method == METHOD_CP else 1,
                             0.0, 0, PUBLISH_NONE, objective="sla", sla_factor=0.85))
    return specs


EXPERIMENTS = {
    "smoke": smoke,
    "E1": e1_convergence,
    "E1b": e1b_convergence_hard,
    "E2": e2_tractability,
    "E3": e3_visibility,
    "E3b": e3b_visibility,
    "E4": e4_main_evaluation,
    "E5": e5_time_limits,
    "E6": e6_value_of_stochastic_solution,
}


def build_specs(name: str) -> list[RunSpec]:
    try:
        return EXPERIMENTS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment '{name}'; available: {sorted(EXPERIMENTS)}."
        ) from exc
