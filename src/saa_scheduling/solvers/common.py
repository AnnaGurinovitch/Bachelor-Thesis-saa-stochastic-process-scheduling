"""
Shared model-building helpers (src/saa_scheduling/solvers/common.py).

Both formulations need the same horizon, the same scaled objective weights and
the same way of turning a solved model into a Plan, so those parts live here.
"""

from __future__ import annotations

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.execution.plan import Plan
from saa_scheduling.sampling.scenario import Scenario
from saa_scheduling.solvers.base import OBJECTIVE_SCALE, TARDINESS_WEIGHT


def compute_horizon(instance: ProblemInstance, scenarios: tuple[Scenario, ...]) -> int:
    """Upper bound on any completion time: every task at its longest duration.

    The bound is derived from the sampled durations rather than from the
    distributions, so it holds whatever family was used.
    """
    return sum(
        max(scenario.max_duration(task_id) for scenario in scenarios)
        for task_id in instance.tasks()
    )


def scaled_weight(visibility_weight: float) -> int:
    """The visibility weight as the integer CP-SAT works with."""
    if visibility_weight < 0:
        raise ValueError(f"visibility weight must not be negative, got {visibility_weight}.")
    return round(visibility_weight * OBJECTIVE_SCALE)


def unscale(objective: float, n_scenarios: int) -> float:
    """Turns the solver's integer objective back into a per-scenario average."""
    return objective / (OBJECTIVE_SCALE * n_scenarios)


def order_from_start_times(
    instance: ProblemInstance, start_times: dict[str, int]
) -> tuple[str, ...]:
    """The committed order implied by the start times of one scenario.

    Sorting by start time always respects precedence, because every duration is
    at least one minute, and ties are broken by task id so the result is
    deterministic.
    """
    return tuple(sorted(instance.tasks(), key=lambda task: (start_times[task], task)))


def build_plan(
    instance: ProblemInstance,
    modes: dict[str, str],
    planned_starts: dict[str, int],
    reference_start_times: dict[str, int],
) -> Plan:
    plan = Plan.create(
        modes=modes,
        order=order_from_start_times(instance, reference_start_times),
        planned_starts=planned_starts,
    )
    plan.validate(instance)
    return plan


def add_objective(
    model,
    instance: ProblemInstance,
    start: dict,
    end: dict,
    planned: dict,
    n_scenarios: int,
    horizon: int,
    weight: int,
    due_dates: dict[str, int] | None,
) -> None:
    """Sets the SAA objective shared by both formulations.

    Summed over scenarios (the average up to the constant 1/N):
        makespan
      + lambda * deviation of the start times from the plan
      + TARDINESS_WEIGHT * tardiness of every case beyond its due date
    """
    tasks = instance.tasks()
    terms = []
    for index in range(n_scenarios):
        makespan = model.NewIntVar(0, horizon, f"makespan_s{index}")
        model.AddMaxEquality(makespan, [end[(task, index)] for task in tasks])
        terms.append(OBJECTIVE_SCALE * makespan)

        if due_dates is not None:
            for case in instance.cases:
                completion = model.NewIntVar(0, horizon, f"completion_{case.case_id}_s{index}")
                model.AddMaxEquality(
                    completion, [end[(task, index)] for task in instance.final_tasks(case)]
                )
                tardiness = model.NewIntVar(0, horizon, f"tardiness_{case.case_id}_s{index}")
                model.Add(tardiness >= completion - due_dates[case.case_id])
                terms.append(OBJECTIVE_SCALE * TARDINESS_WEIGHT * tardiness)

        if weight:
            terms.append(weight * sum(start[(task, index)] - planned[task] for task in tasks))

    model.Minimize(sum(terms))
