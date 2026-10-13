"""
Deterministic baselines (src/evaluation/baselines.py).

Two ways of scheduling with a single duration value per activity:

  DET  plan with the mean duration of every activity
  ROB  plan with a high quantile of every duration distribution

Both solve the very same model as SAA, but with one scenario instead of a
sample, so any difference in performance comes from the duration information
used and not from a different scheduling method.

Each baseline is evaluated twice. Published with the start times of its own
planning scenario ("own"), it is what a planner would hand out, and its
visibility is meaningful. Published without start times ("none"), it only
fixes modes and order, which is the fair comparison with an SAA plan that
promises no times either.
"""

from __future__ import annotations

from collections.abc import Sequence

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.distributions.activity_durations import ActivityDurationModel
from saa_scheduling.execution.plan import Plan
from saa_scheduling.execution.simulator import execute
from saa_scheduling.sampling.scenario import Scenario
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator
from saa_scheduling.solvers.base import SolveResult, Solver

#: Quantile used by the robust baseline.
ROBUST_QUANTILE = 0.9


def planning_scenario(
    instance: ProblemInstance,
    duration_model: ActivityDurationModel | Sequence[ActivityDurationModel],
    quantile: float | None,
) -> Scenario:
    """The single scenario a baseline plans with: means, or a quantile."""
    return ScenarioGenerator(instance, duration_model, seed=0).deterministic_scenario(quantile)


def solve_baseline(
    instance: ProblemInstance,
    duration_model: ActivityDurationModel | Sequence[ActivityDurationModel],
    solver: Solver,
    quantile: float | None = None,
    due_dates: dict[str, int] | None = None,
) -> tuple[SolveResult, Scenario]:
    """Solves the single-scenario problem; returns the result and its scenario.

    quantile: None for the mean-based baseline (DET), or a probability for the
        robust baseline (ROB).
    due_dates: the same SLA the SAA plan is optimised for, if any.
    """
    scenario = planning_scenario(instance, duration_model, quantile)
    result = solver.solve(instance, (scenario,), visibility_weight=0.0, due_dates=due_dates)
    return result, scenario


def with_own_start_times(instance: ProblemInstance, plan: Plan, scenario: Scenario) -> Plan:
    """Publishes the start times the plan has in its own planning scenario."""
    execution = execute(instance, plan, scenario)
    published = Plan.create(
        modes=dict(plan.modes), order=plan.order, planned_starts=dict(execution.starts)
    )
    published.validate(instance)
    return published
