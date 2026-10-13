"""
Core process model (src/saa_scheduling/basics/process.py).

Defines the structural building blocks that every solver depends on: an
execution mode, an activity, a reusable process template (the BPMN diagram)
and a case (one running instance of a template). The module knows nothing
about durations, capacities or solvers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from saa_scheduling.basics.exceptions import (
    DuplicateActivityError,
    PrecedenceCycleError,
    UnknownActivityError,
    UnknownCaseTaskError,
    UnknownModeError,
)

CASE_TASK_SEPARATOR = ":"


@dataclass(frozen=True)
class Mode:
    """One way of executing an activity.

    name: identifier, unique within its activity, e.g. "junior".
    role: the resource pool one unit of which the mode occupies.
    duration_factor: the activity's base duration is multiplied by this
        factor in this mode. A senior officer working faster than a junior
        analyst is modelled as a factor below one.
    """

    name: str
    role: str
    duration_factor: float = 1.0


@dataclass(frozen=True)
class Activity:
    """One task of a process template.

    name: unique identifier within its template.
    modes: the available execution modes; exactly one is chosen per plan.
    predecessors: activities that must finish before this one may start.
        Several predecessors model an AND-join.
    """

    name: str
    modes: tuple[Mode, ...]
    predecessors: tuple[str, ...] = field(default_factory=tuple)

    def mode(self, mode_name: str) -> Mode:
        for mode in self.modes:
            if mode.name == mode_name:
                return mode
        raise UnknownModeError(
            f"Activity '{self.name}' has no mode '{mode_name}'."
        )

    @property
    def roles(self) -> frozenset[str]:
        return frozenset(mode.role for mode in self.modes)


class ProcessTemplate:
    """An immutable, validated process definition (the diagram itself)."""

    def __init__(self, name: str, activities: list[Activity]):
        self._name = name
        self._activities: dict[str, Activity] = {}
        for activity in activities:
            if activity.name in self._activities:
                raise DuplicateActivityError(
                    f"Activity '{activity.name}' is defined twice in template '{name}'."
                )
            if not activity.modes:
                raise UnknownModeError(
                    f"Activity '{activity.name}' has no execution mode."
                )
            self._activities[activity.name] = activity
        self._validate_predecessors_exist()
        self._topological_order = self._compute_topological_order()

    @property
    def name(self) -> str:
        return self._name

    @property
    def activity_names(self) -> tuple[str, ...]:
        return tuple(self._activities.keys())

    @property
    def roles(self) -> frozenset[str]:
        return frozenset().union(*(a.roles for a in self._activities.values()))

    def activity(self, activity_name: str) -> Activity:
        try:
            return self._activities[activity_name]
        except KeyError as exc:
            raise UnknownActivityError(
                f"Activity '{activity_name}' is not part of template '{self._name}'."
            ) from exc

    def predecessors_of(self, activity_name: str) -> tuple[str, ...]:
        return self.activity(activity_name).predecessors

    def topological_order(self) -> tuple[str, ...]:
        return self._topological_order

    def _validate_predecessors_exist(self) -> None:
        for activity in self._activities.values():
            for predecessor in activity.predecessors:
                if predecessor not in self._activities:
                    raise UnknownActivityError(
                        f"Activity '{activity.name}' references unknown "
                        f"predecessor '{predecessor}'."
                    )

    def _compute_topological_order(self) -> tuple[str, ...]:
        in_degree = {name: 0 for name in self._activities}
        successors: dict[str, list[str]] = {name: [] for name in self._activities}
        for activity in self._activities.values():
            for predecessor in activity.predecessors:
                successors[predecessor].append(activity.name)
                in_degree[activity.name] += 1

        ready = deque(name for name, degree in in_degree.items() if degree == 0)
        order: list[str] = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for successor in successors[current]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready.append(successor)

        if len(order) != len(self._activities):
            unresolved = sorted(set(self._activities) - set(order))
            raise PrecedenceCycleError(
                f"Template '{self._name}' contains a precedence cycle involving: {unresolved}."
            )
        return tuple(order)


@dataclass(frozen=True)
class Case:
    """One running instance of a ProcessTemplate, e.g. case 'C0'."""

    case_id: str
    template: ProcessTemplate

    def task_id(self, activity_name: str) -> str:
        self.template.activity(activity_name)
        return f"{self.case_id}{CASE_TASK_SEPARATOR}{activity_name}"

    def tasks(self) -> tuple[str, ...]:
        return tuple(self.task_id(name) for name in self.template.topological_order())

    def activity_of(self, task_id: str) -> Activity:
        return self.template.activity(self.activity_name(task_id))

    def predecessors_of(self, task_id: str) -> tuple[str, ...]:
        activity_name = self.activity_name(task_id)
        return tuple(self.task_id(p) for p in self.template.predecessors_of(activity_name))

    def activity_name(self, task_id: str) -> str:
        prefix = f"{self.case_id}{CASE_TASK_SEPARATOR}"
        if not task_id.startswith(prefix):
            raise UnknownCaseTaskError(
                f"Task '{task_id}' does not belong to case '{self.case_id}'."
            )
        return task_id[len(prefix):]
