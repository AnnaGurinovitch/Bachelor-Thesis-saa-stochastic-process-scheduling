"""
Due dates (src/evaluation/due_dates.py).

A service level agreement promises every case a decision within a fixed time.
Here all cases are released at time zero and share one due date, derived from
a reference makespan: the naive plan (cases one after the other, junior modes)
executed with mean durations. A factor below one gives a tight SLA, a factor
above one a loose one.

The reference depends only on the means, so it is the same for a homogeneous
instance and for a mixed-risk instance with the same means.
"""

from __future__ import annotations

from collections.abc import Sequence

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.distributions.activity_durations import ActivityDurationModel
from saa_scheduling.execution.plan import Plan
from saa_scheduling.execution.simulator import execute
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator


def reference_makespan(
    instance: ProblemInstance, models: Sequence[ActivityDurationModel]
) -> int:
    """Makespan of the naive plan under mean durations."""
    scenario = ScenarioGenerator(instance, models, seed=0).deterministic_scenario()
    order = list(instance.tasks())
    naive = Plan.create(
        modes={task: instance.modes_of(task)[0].name for task in order},
        order=order,
        planned_starts={task: 0 for task in order},
    )
    return execute(instance, naive, scenario).makespan


def common_due_dates(
    instance: ProblemInstance,
    models: Sequence[ActivityDurationModel],
    factor: float,
) -> dict[str, int]:
    """The same due date for every case: factor times the reference makespan."""
    due = round(factor * reference_makespan(instance, models))
    return {case.case_id: due for case in instance.cases}
