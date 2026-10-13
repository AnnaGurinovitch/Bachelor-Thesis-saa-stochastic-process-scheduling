"""
Publishing start times (src/saa_scheduling/execution/publication.py).

A plan from SAA with lambda = 0 fixes modes and order but promises no start
times. This module turns such a plan into a visible one after the fact: the
plan is simulated on the planning scenarios, and every task receives the time
by which it has started in a share q of them.

    q = 0.5   the median start time: an optimistic promise
    q = 0.9   a start time that held in nine out of ten planning scenarios

Only the planning scenarios are used, never the hold-out sample, so the
published times rely on no information a planner would not have. Because a
published time also delays tasks that could have started earlier, the
realised on-time share is lower than q; the simulator measures by how much.
"""

from __future__ import annotations

import math

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.execution.plan import Plan
from saa_scheduling.sampling.scenario import Scenario


def empirical_quantile(values: list[int], probability: float) -> int:
    """The smallest observed value that at least `probability` of values do not exceed."""
    if not values:
        raise ValueError("empirical_quantile needs at least one value.")
    if not 0.0 < probability <= 1.0:
        raise ValueError(f"probability must lie in (0, 1], got {probability}.")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def publish_quantile_times(
    instance: ProblemInstance,
    plan: Plan,
    scenarios: tuple[Scenario, ...],
    probability: float,
) -> Plan:
    """The same modes and order, with start times published at quantile q.

    The times are fixed task by task in the plan's order, executing all
    scenarios in lockstep. When a task's time is published, the times of all
    earlier tasks are already in force, including the waiting they cause, so
    every promise accounts for the promises made before it. The execution rule
    is exactly the simulator's: a task starts at the latest of its published
    time, the end of its predecessors and the release of a unit of its pool.
    """
    plan.validate(instance)
    free_at = [
        {role: [0] * capacity for role, capacity in instance.capacities.items()}
        for _ in scenarios
    ]
    ends: list[dict[str, int]] = [{} for _ in scenarios]
    published: dict[str, int] = {}

    for task in plan.order:
        mode_name = plan.mode_of(task)
        role = instance.mode_of(task, mode_name).role
        predecessors = instance.predecessors_of(task)

        ready = []
        for index in range(len(scenarios)):
            units = free_at[index][role]
            earliest_unit = min(units)
            predecessors_done = max((ends[index][p] for p in predecessors), default=0)
            ready.append(max(earliest_unit, predecessors_done))

        published[task] = empirical_quantile(ready, probability)

        for index, scenario in enumerate(scenarios):
            units = free_at[index][role]
            unit = min(range(len(units)), key=lambda i: units[i])
            start = max(published[task], ready[index])
            ends[index][task] = start + scenario.duration_of(task, mode_name)
            units[unit] = ends[index][task]

    result = Plan.create(modes=dict(plan.modes), order=plan.order, planned_starts=published)
    result.validate(instance)
    return result


def without_start_times(plan: Plan) -> Plan:
    """The same modes and order, with no start time promised (all zero)."""
    return Plan.create(
        modes=dict(plan.modes),
        order=plan.order,
        planned_starts={task: 0 for task in plan.order},
    )
