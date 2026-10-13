"""
Experiment runner (src/evaluation/runner.py).

Executes RunSpec records one by one: build the instance, draw the scenarios,
solve, publish start times, then execute the plan on an independent hold-out
sample and record everything in one CSV row. Every plan is also stored as a
small JSON file, so it can be re-evaluated or drawn as a Gantt chart later.

Properties that matter for overnight runs:
  * results are written immediately, so nothing is lost if the machine stops;
  * finished rows are skipped on a restart;
  * rows that differ only in how start times are published share one solve;
  * a failed or timed-out solve is recorded with its status instead of ending
    the experiment, and larger case counts of that configuration are skipped.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import replace

from evaluation.baselines import ROBUST_QUANTILE, solve_baseline, with_own_start_times
from evaluation.due_dates import common_due_dates
from evaluation.holdout import HOLDOUT_SIZE, holdout_scenarios
from evaluation.experiments import (
    METHOD_CP,
    METHOD_DET,
    METHOD_MSEQCT,
    METHOD_ROB,
    RunSpec,
    quantile_of,
)
from evaluation.results import (
    PUBLISH_NONE,
    PUBLISH_OPTIMISED,
    PUBLISH_OWN,
    ResultsFile,
    RunRow,
)
from saa_scheduling.basics.instances import ProblemInstance, build_instance
from saa_scheduling.distributions.duration_profiles import build_case_models
from saa_scheduling.execution.plan import Plan
from saa_scheduling.execution.publication import publish_quantile_times, without_start_times
from saa_scheduling.execution.simulator import evaluate
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator
from saa_scheduling.solvers.base import SolveResult, SolverConfig
from saa_scheduling.solvers.saa_cp import SaaCpSolver
from saa_scheduling.solvers.saa_mseqct import SaaMseqctSolver

SOLVERS = {METHOD_CP: SaaCpSolver, METHOD_MSEQCT: SaaMseqctSolver}
_SOLVED = ("OPTIMAL", "FEASIBLE")


class Runner:
    """Runs specs against a results file, skipping what is already done."""

    def __init__(
        self,
        results: ResultsFile,
        config: SolverConfig,
        holdout_size: int = HOLDOUT_SIZE,
        plan_directory: str | None = None,
        verbose: bool = True,
    ):
        self._results = results
        self._config = config
        self._holdout_size = holdout_size
        self._plan_directory = plan_directory or os.path.join(
            os.path.dirname(results.path) or ".", "plans"
        )
        self._verbose = verbose
        self._holdout_cache: dict[tuple, tuple] = {}
        self._solve_cache: dict[tuple, tuple] = {}

    def run_all(self, specs: list[RunSpec]) -> None:
        # configuration -> smallest number of cases that already failed
        failed_from: dict[tuple, int] = {}
        for index, spec in enumerate(specs, start=1):
            limit = failed_from.get(self._configuration(spec))
            if limit is not None and spec.cases >= limit:
                self._say(
                    f"[{index}/{len(specs)}] skip, this configuration already failed "
                    f"at {limit} cases: {spec.label}"
                )
                continue
            row = self._maybe_run(spec, index, len(specs))
            if row is not None and not row.status.startswith(_SOLVED):
                configuration = self._configuration(spec)
                failed_from[configuration] = min(
                    spec.cases, failed_from.get(configuration, spec.cases)
                )

    # ------------------------------------------------------------------
    def _maybe_run(self, spec: RunSpec, index: int, total: int) -> RunRow | None:
        if self._empty_row(spec, status="").key in self._results:
            self._say(f"[{index}/{total}] done already {spec.label}")
            return None

        self._say(f"[{index}/{total}] {spec.label}")
        try:
            row = self._run(spec)
        except MemoryError:
            row = self._empty_row(spec, status="OUT_OF_MEMORY")
        except Exception as error:  # a failed run must not stop the night
            row = self._empty_row(spec, status="ERROR", note=f"{type(error).__name__}: {error}")
            traceback.print_exc()
        self._results.append(row)
        self._say(
            f"    -> {row.status}"
            + (f", makespan {row.oos_makespan:.1f}" if row.oos_makespan else "")
            + (
                f", on time {row.oos_on_time_share:.2f}, visible {row.oos_visibility:.2f}"
                if row.oos_on_time_share is not None
                else ""
            )
            + f", {row.wall_seconds:.1f}s"
        )
        return row

    def _run(self, spec: RunSpec) -> RunRow:
        instance, result, publication_input = self._solve(spec)
        if not result.solved:
            return self._empty_row(
                spec,
                status=result.status,
                wall_seconds=result.wall_seconds,
                deterministic_seconds=result.deterministic_seconds,
                n_variables=result.n_variables,
                n_constraints=result.n_constraints,
            )

        plan = self._publish(spec, instance, result.plan, publication_input)
        due_dates = self._due_dates(spec, instance)
        realised = evaluate(
            instance, plan, self._holdout(spec), spec.visibility_weight, due_dates
        )

        return RunRow(
            experiment=spec.experiment,
            process=spec.process,
            cases=spec.cases,
            capacities=spec.capacity_label,
            level=spec.level,
            method=spec.method,
            n_scenarios=spec.n_scenarios,
            visibility_weight=spec.visibility_weight,
            seed=spec.seed,
            status=result.status,
            objective_in_sample=result.objective,
            bound=result.bound,
            gap=result.gap,
            wall_seconds=result.wall_seconds,
            deterministic_seconds=result.deterministic_seconds,
            n_variables=result.n_variables,
            n_constraints=result.n_constraints,
            oos_makespan=realised.mean_makespan,
            oos_makespan_std=realised.std_makespan,
            oos_visibility=realised.visibility,
            oos_mean_delay=realised.mean_total_delay,
            oos_max_delay=realised.max_delay,
            publish=spec.publish,
            oos_visibility_5=realised.visibility_at[5],
            oos_visibility_15=realised.visibility_at[15],
            oos_on_time_share=realised.on_time_share_at[0],
            oos_on_time_share_5=realised.on_time_share_at[5],
            oos_on_time_share_15=realised.on_time_share_at[15],
            plan_file=self._save_plan(spec, plan),
            time_limit=spec.time_limit_label,
            objective=spec.objective,
            sla_factor=spec.sla_label,
            due_date=None if due_dates is None else next(iter(due_dates.values())),
            oos_mean_case_completion=realised.mean_case_completion,
            oos_total_tardiness=realised.mean_total_tardiness,
            oos_sla_violation=realised.sla_violation_rate,
        )

    # ------------------------------------------------------------------
    def _solve(self, spec: RunSpec) -> tuple[ProblemInstance, SolveResult, tuple]:
        """Solves once per solve key; later publications reuse the result."""
        if spec.solve_key in self._solve_cache:
            return self._solve_cache[spec.solve_key]

        instance = build_instance(spec.process, spec.cases, dict(spec.capacities))
        duration_model = build_case_models(spec.level, spec.cases)
        due_dates = self._due_dates(spec, instance)
        config = (
            self._config
            if spec.max_seconds is None
            else replace(self._config, max_seconds=spec.max_seconds)
        )

        if spec.method in (METHOD_DET, METHOD_ROB):
            quantile = ROBUST_QUANTILE if spec.method == METHOD_ROB else None
            result, scenario = solve_baseline(
                instance, duration_model, SaaCpSolver(config), quantile, due_dates
            )
            publication_input = (scenario,)
        else:
            scenarios = ScenarioGenerator(instance, duration_model, seed=spec.seed).generate(
                spec.n_scenarios
            )
            result = SOLVERS[spec.method](config).solve(
                instance, scenarios, spec.visibility_weight, due_dates
            )
            publication_input = scenarios

        self._solve_cache = {spec.solve_key: (instance, result, publication_input)}
        return instance, result, publication_input

    @staticmethod
    def _due_dates(spec: RunSpec, instance: ProblemInstance) -> dict[str, int] | None:
        if spec.objective == "makespan":
            return None
        if spec.objective == "sla" and spec.sla_factor is not None:
            models = build_case_models(spec.level, spec.cases)
            return common_due_dates(instance, models, spec.sla_factor)
        raise ValueError(f"Unknown objective '{spec.objective}' for {spec.label}.")

    @staticmethod
    def _publish(spec: RunSpec, instance: ProblemInstance, plan: Plan, scenarios: tuple) -> Plan:
        """Sets the start times the row is evaluated with."""
        if spec.publish == PUBLISH_NONE:
            return without_start_times(plan)
        if spec.publish == PUBLISH_OPTIMISED:
            return plan
        if spec.publish == PUBLISH_OWN:
            return with_own_start_times(instance, plan, scenarios[0])
        probability = quantile_of(spec.publish)
        if probability is None:
            raise ValueError(f"Unknown publication '{spec.publish}'.")
        return publish_quantile_times(instance, plan, scenarios, probability)

    def _holdout(self, spec: RunSpec) -> tuple:
        key = (spec.process, spec.cases, spec.capacity_label, spec.level)
        if key not in self._holdout_cache:
            self._holdout_cache = {
                key: holdout_scenarios(
                    spec.process, spec.cases, dict(spec.capacities), spec.level,
                    self._holdout_size,
                )
            }
        return self._holdout_cache[key]

    def _save_plan(self, spec: RunSpec, plan: Plan) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", spec.label) + ".json"
        path = os.path.join(self._plan_directory, spec.experiment, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(
                {
                    "spec": spec.label,
                    "modes": dict(plan.modes),
                    "order": list(plan.order),
                    "planned_starts": dict(plan.planned_starts),
                },
                handle,
                indent=1,
            )
        return path

    @staticmethod
    def _configuration(spec: RunSpec) -> tuple:
        """Everything but the number of cases, which is the size dimension."""
        return (
            spec.experiment, spec.process, spec.capacity_label, spec.level,
            spec.method, spec.n_scenarios, spec.visibility_weight, spec.publish,
            spec.max_seconds, spec.objective, spec.sla_factor,
        )

    def _empty_row(self, spec: RunSpec, status: str, note: str = "", **kwargs) -> RunRow:
        defaults = dict(wall_seconds=0.0, deterministic_seconds=0.0, n_variables=0, n_constraints=0)
        defaults.update(kwargs)
        return RunRow(
            experiment=spec.experiment, process=spec.process, cases=spec.cases,
            capacities=spec.capacity_label, level=spec.level, method=spec.method,
            n_scenarios=spec.n_scenarios, visibility_weight=spec.visibility_weight,
            seed=spec.seed, status=status, objective_in_sample=None, bound=None, gap=None,
            oos_makespan=None, oos_makespan_std=None, oos_visibility=None,
            oos_mean_delay=None, oos_max_delay=None, note=note, publish=spec.publish,
            time_limit=spec.time_limit_label, objective=spec.objective,
            sla_factor=spec.sla_label, **defaults,
        )

    def _say(self, message: str) -> None:
        if self._verbose:
            print(message, flush=True)
