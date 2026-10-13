"""
Plan execution (src/saa_scheduling/execution/simulator.py).

Executes a plan under one realised scenario and measures the outcome. This
module is the independent reference of the whole pipeline: every plan, whether
it comes from SAA-MSEQCT, SAA-CP or a baseline, is judged by running it here,
never by the objective value its solver reported.

Execution rule: tasks claim their resource in the committed order of the plan.
A task starts at the latest of three times -- its planned start time, the
completion of all its predecessors, and the moment a unit of its resource pool
becomes free.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, stdev
from types import MappingProxyType

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.execution.plan import Plan
from saa_scheduling.sampling.scenario import Scenario

#: Delays (in minutes) up to which a start still counts as "on time" in the
#: tolerant visibility metrics. Zero is the strict definition.
VISIBILITY_TOLERANCES = (0, 5, 15)


@dataclass(frozen=True)
class Execution:
    """The realised schedule of one plan under one scenario."""

    starts: MappingProxyType
    ends: MappingProxyType
    makespan: int

    def delay_of(self, task_id: str, plan: Plan) -> int:
        return self.starts[task_id] - plan.planned_start_of(task_id)

    def total_delay(self, plan: Plan) -> int:
        return sum(self.delay_of(task_id, plan) for task_id in self.starts)

    def is_visible(self, plan: Plan, tolerance: int = 0) -> bool:
        """True if every task started at most `tolerance` minutes after its plan."""
        return all(self.delay_of(task_id, plan) <= tolerance for task_id in self.starts)

    def case_completions(self, instance: ProblemInstance) -> dict[str, int]:
        """Completion time of every case: the end of its final tasks."""
        return {
            case.case_id: max(self.ends[task] for task in instance.final_tasks(case))
            for case in instance.cases
        }

    def on_time_share(self, plan: Plan, tolerance: int = 0) -> float:
        """Share of tasks that started at most `tolerance` minutes late."""
        on_time = sum(
            1 for task_id in self.starts if self.delay_of(task_id, plan) <= tolerance
        )
        return on_time / len(self.starts)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated outcome of a plan over a set of scenarios.

    visibility: share of scenarios in which every task started on time.
    visibility_at: the same with a tolerance of k minutes, per tolerance.
    on_time_share_at: share of tasks that started within k minutes, averaged
        over the scenarios, per tolerance. Less strict than visibility: one late
        task lowers it by one task, not by a whole scenario.
    punctuality: per task, the share of scenarios in which it started on time.
    """

    n_scenarios: int
    mean_makespan: float
    std_makespan: float
    visibility: float
    mean_total_delay: float
    max_delay: int
    punctuality: MappingProxyType
    objective: float
    visibility_at: MappingProxyType
    on_time_share_at: MappingProxyType
    mean_case_completion: float = 0.0
    #: only with due dates: mean over scenarios of the summed case tardiness
    mean_total_tardiness: float | None = None
    #: only with due dates: share of (case, scenario) pairs that miss their due date
    sla_violation_rate: float | None = None

    def as_row(self) -> dict:
        return {
            "n_scenarios": self.n_scenarios,
            "mean_makespan": self.mean_makespan,
            "std_makespan": self.std_makespan,
            "visibility": self.visibility,
            "mean_total_delay": self.mean_total_delay,
            "max_delay": self.max_delay,
            "objective": self.objective,
            **{f"visibility_{k}": v for k, v in self.visibility_at.items()},
            **{f"on_time_share_{k}": v for k, v in self.on_time_share_at.items()},
        }


def execute(instance: ProblemInstance, plan: Plan, scenario: Scenario) -> Execution:
    """Runs one plan under one scenario and returns the realised schedule."""
    plan.validate(instance)

    free_at: dict[str, list[int]] = {
        role: [0] * capacity for role, capacity in instance.capacities.items()
    }
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}

    for task_id in plan.order:
        mode_name = plan.mode_of(task_id)
        mode = instance.mode_of(task_id, mode_name)
        units = free_at[mode.role]
        unit_index = min(range(len(units)), key=lambda i: units[i])

        ready = plan.planned_start_of(task_id)
        for predecessor in instance.predecessors_of(task_id):
            ready = max(ready, ends[predecessor])
        start = max(ready, units[unit_index])

        starts[task_id] = start
        ends[task_id] = start + scenario.duration_of(task_id, mode_name)
        units[unit_index] = ends[task_id]

    return Execution(
        starts=MappingProxyType(starts),
        ends=MappingProxyType(ends),
        makespan=max(ends.values()),
    )


def evaluate(
    instance: ProblemInstance,
    plan: Plan,
    scenarios: tuple[Scenario, ...],
    visibility_weight: float = 0.0,
    due_dates: dict[str, int] | None = None,
) -> EvaluationResult:
    """Executes a plan on every scenario and aggregates the metrics.

    due_dates: optional due date (SLA) per case id; enables the tardiness and
        violation metrics.
    """
    if not scenarios:
        raise ValueError("evaluate needs at least one scenario.")

    executions = [execute(instance, plan, scenario) for scenario in scenarios]
    makespans = [execution.makespan for execution in executions]
    total_delays = [execution.total_delay(plan) for execution in executions]
    on_time = {task_id: 0 for task_id in instance.tasks()}
    max_delay = 0

    for execution in executions:
        for task_id in instance.tasks():
            delay = execution.delay_of(task_id, plan)
            max_delay = max(max_delay, delay)
            if delay == 0:
                on_time[task_id] += 1

    n = len(scenarios)
    completions = [execution.case_completions(instance) for execution in executions]
    mean_case_completion = fmean(
        value for completion in completions for value in completion.values()
    )
    total_tardiness = violation_rate = None
    if due_dates is not None:
        tardiness = [
            [max(0, completion[case] - due_dates[case]) for case in completion]
            for completion in completions
        ]
        total_tardiness = fmean(sum(values) for values in tardiness)
        violation_rate = fmean(
            1.0 if value > 0 else 0.0 for values in tardiness for value in values
        )

    objective = fmean(
        makespan + visibility_weight * delay
        for makespan, delay in zip(makespans, total_delays)
    )
    return EvaluationResult(
        n_scenarios=n,
        mean_makespan=fmean(makespans),
        std_makespan=stdev(makespans) if n > 1 else 0.0,
        visibility=sum(1 for e in executions if e.is_visible(plan)) / n,
        mean_total_delay=fmean(total_delays),
        max_delay=max_delay,
        punctuality=MappingProxyType(
            {task_id: count / n for task_id, count in on_time.items()}
        ),
        objective=objective,
        visibility_at=MappingProxyType(
            {
                k: sum(1 for e in executions if e.is_visible(plan, k)) / n
                for k in VISIBILITY_TOLERANCES
            }
        ),
        on_time_share_at=MappingProxyType(
            {
                k: fmean(e.on_time_share(plan, k) for e in executions)
                for k in VISIBILITY_TOLERANCES
            }
        ),
        mean_case_completion=mean_case_completion,
        mean_total_tardiness=total_tardiness,
        sla_violation_rate=violation_rate,
    )
