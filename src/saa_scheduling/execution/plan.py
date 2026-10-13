"""
Plan (src/saa_scheduling/execution/plan.py).

A Plan is what a formulation returns and what a process participant would be
handed: the mode of every task, the committed order in which tasks claim their
resource, and the planned start time of every task. These are exactly the
first-stage decisions of the two-stage model; nothing in a Plan depends on a
scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.execution.exceptions import InfeasiblePlanError


@dataclass(frozen=True)
class Plan:
    """Modes, committed order and planned start times of all tasks."""

    modes: MappingProxyType
    order: tuple[str, ...]
    planned_starts: MappingProxyType

    @staticmethod
    def create(
        modes: dict[str, str],
        order: list[str] | tuple[str, ...],
        planned_starts: dict[str, int],
    ) -> "Plan":
        return Plan(
            modes=MappingProxyType(dict(modes)),
            order=tuple(order),
            planned_starts=MappingProxyType(dict(planned_starts)),
        )

    def mode_of(self, task_id: str) -> str:
        return self.modes[task_id]

    def planned_start_of(self, task_id: str) -> int:
        return self.planned_starts[task_id]

    def validate(self, instance: ProblemInstance) -> None:
        """Checks that the plan covers the instance and respects precedence."""
        tasks = set(instance.tasks())
        if set(self.order) != tasks or len(self.order) != len(tasks):
            raise InfeasiblePlanError("The plan's order does not list every task exactly once.")
        if set(self.modes) != tasks:
            raise InfeasiblePlanError("The plan does not assign a mode to every task.")
        if set(self.planned_starts) != tasks:
            raise InfeasiblePlanError("The plan does not give a planned start for every task.")

        position = {task_id: index for index, task_id in enumerate(self.order)}
        for task_id in self.order:
            instance.mode_of(task_id, self.modes[task_id])  # raises if unknown
            if self.planned_starts[task_id] < 0:
                raise InfeasiblePlanError(
                    f"Planned start of '{task_id}' is negative."
                )
            for predecessor in instance.predecessors_of(task_id):
                if position[predecessor] > position[task_id]:
                    raise InfeasiblePlanError(
                        f"'{predecessor}' must precede '{task_id}' but comes later in the order."
                    )
