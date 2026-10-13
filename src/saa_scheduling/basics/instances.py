"""
Process models and problem instances (src/saa_scheduling/basics/instances.py).

Defines the three process models P1--P3 of the thesis and the ProblemInstance
that combines a process model with a number of concurrent cases and the
capacities of the resource pools.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType

from saa_scheduling.basics.exceptions import UnknownRoleError
from saa_scheduling.basics.process import Activity, Case, Mode, ProcessTemplate

JUNIOR = "Junior"
SENIOR = "Senior"

#: A senior officer needs this share of the junior duration for a junior task.
SENIOR_SPEEDUP = 0.6

MODE_JUNIOR = "junior"
MODE_SENIOR = "senior"


def _junior_task(name: str, predecessors: tuple[str, ...] = ()) -> Activity:
    """A task normally done by a junior analyst; a senior may take it over."""
    return Activity(
        name=name,
        modes=(
            Mode(MODE_JUNIOR, JUNIOR, 1.0),
            Mode(MODE_SENIOR, SENIOR, SENIOR_SPEEDUP),
        ),
        predecessors=predecessors,
    )


def _senior_task(name: str, predecessors: tuple[str, ...] = ()) -> Activity:
    """A task that only a senior officer may perform."""
    return Activity(
        name=name,
        modes=(Mode(MODE_SENIOR, SENIOR, 1.0),),
        predecessors=predecessors,
    )


def build_p1() -> ProcessTemplate:
    """Base loan process: intake, AND(risk, document check), scoring, decision."""
    return ProcessTemplate(
        name="P1",
        activities=[
            _junior_task("Application Intake"),
            _senior_task("Risk Assessment", ("Application Intake",)),
            _junior_task("Document Check", ("Application Intake",)),
            _junior_task("Credit Scoring", ("Risk Assessment", "Document Check")),
            _senior_task("Loan Decision", ("Credit Scoring",)),
        ],
    )


def build_p2() -> ProcessTemplate:
    """Wider parallel block: credit scoring runs inside the AND-block."""
    return ProcessTemplate(
        name="P2",
        activities=[
            _junior_task("Application Intake"),
            _senior_task("Risk Assessment", ("Application Intake",)),
            _junior_task("Document Check", ("Application Intake",)),
            _junior_task("Credit Scoring", ("Application Intake",)),
            _senior_task(
                "Loan Decision",
                ("Risk Assessment", "Document Check", "Credit Scoring"),
            ),
        ],
    )


def build_p3() -> ProcessTemplate:
    """Nested structure: an inner AND-block inside the document check branch."""
    return ProcessTemplate(
        name="P3",
        activities=[
            _junior_task("Application Intake"),
            _senior_task("Risk Assessment", ("Application Intake",)),
            _junior_task("Document Check", ("Application Intake",)),
            _junior_task("Verify Income", ("Document Check",)),
            _junior_task("Verify Identity", ("Document Check",)),
            _junior_task(
                "Credit Scoring", ("Risk Assessment", "Verify Income", "Verify Identity")
            ),
            _senior_task("Loan Decision", ("Credit Scoring",)),
        ],
    )


PROCESS_BUILDERS = {"P1": build_p1, "P2": build_p2, "P3": build_p3}


def build_process(name: str) -> ProcessTemplate:
    try:
        return PROCESS_BUILDERS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown process model '{name}'; available: {sorted(PROCESS_BUILDERS)}."
        ) from exc


class ProblemInstance:
    """One scheduling instance: a process model, its cases and the capacities.

    The instance owns the task ids used everywhere downstream and answers the
    structural questions a solver needs: which tasks exist, which precede which,
    which modes a task has, and which pairs of tasks are free, i.e. unrelated by
    precedence and therefore candidates for a sequencing decision.
    """

    def __init__(self, template: ProcessTemplate, n_cases: int, capacities: dict[str, int]):
        if n_cases < 1:
            raise ValueError(f"n_cases must be at least 1, got {n_cases}.")
        self._template = template
        self._cases = tuple(Case(f"C{index}", template) for index in range(n_cases))
        self._capacities = MappingProxyType(dict(capacities))
        self._case_of = {t: c for c in self._cases for t in c.tasks()}
        self._tasks = tuple(self._case_of)
        self._validate_capacities()
        self._free_pairs = self._compute_free_pairs()

    @property
    def template(self) -> ProcessTemplate:
        return self._template

    @property
    def cases(self) -> tuple[Case, ...]:
        return self._cases

    @property
    def n_cases(self) -> int:
        return len(self._cases)

    @property
    def capacities(self) -> MappingProxyType:
        return self._capacities

    @property
    def name(self) -> str:
        caps = "-".join(f"{role}{self._capacities[role]}" for role in sorted(self._capacities))
        return f"{self._template.name}_{self.n_cases}cases_{caps}"

    def tasks(self) -> tuple[str, ...]:
        return self._tasks

    def case_of(self, task_id: str) -> Case:
        return self._case_of[task_id]

    def activity_of(self, task_id: str):
        return self._case_of[task_id].activity_of(task_id)

    def activity_name_of(self, task_id: str) -> str:
        return self._case_of[task_id].activity_name(task_id)

    def modes_of(self, task_id: str) -> tuple[Mode, ...]:
        return self.activity_of(task_id).modes

    def mode_of(self, task_id: str, mode_name: str) -> Mode:
        return self.activity_of(task_id).mode(mode_name)

    def predecessors_of(self, task_id: str) -> tuple[str, ...]:
        return self._case_of[task_id].predecessors_of(task_id)

    def case_index(self, task_id: str) -> int:
        return self._cases.index(self._case_of[task_id])

    def final_tasks(self, case: Case) -> tuple[str, ...]:
        """Tasks of a case that no other task of the case waits for.

        The case is complete when all of them are complete, so its completion
        time is the maximum of their end times.
        """
        tasks = case.tasks()
        has_successor = {p for task in tasks for p in case.predecessors_of(task)}
        return tuple(task for task in tasks if task not in has_successor)

    def capacity_of(self, role: str) -> int:
        try:
            return self._capacities[role]
        except KeyError as exc:
            raise UnknownRoleError(
                f"No capacity given for role '{role}' in instance '{self.name}'."
            ) from exc

    def free_pairs(self) -> tuple[tuple[str, str], ...]:
        """Task pairs that are not related by precedence, in a fixed order."""
        return self._free_pairs

    def conflicting_pairs(self) -> tuple[tuple[str, str], ...]:
        """Free pairs that could claim the same resource pool.

        Only these pairs need a sequencing commitment: two tasks whose possible
        roles are disjoint can never block each other, whatever mode is chosen.
        """
        return tuple(
            (first, second)
            for first, second in self._free_pairs
            if self.activity_of(first).roles & self.activity_of(second).roles
        )

    def _validate_capacities(self) -> None:
        for role in self._template.roles:
            if self.capacity_of(role) < 1:
                raise UnknownRoleError(
                    f"Role '{role}' needs a capacity of at least 1."
                )

    def _ancestors(self) -> dict[str, set[str]]:
        ancestors: dict[str, set[str]] = {}
        for case in self._cases:
            for task_id in case.tasks():  # topological order
                direct = case.predecessors_of(task_id)
                closure: set[str] = set(direct)
                for predecessor in direct:
                    closure |= ancestors[predecessor]
                ancestors[task_id] = closure
        return ancestors

    def _compute_free_pairs(self) -> tuple[tuple[str, str], ...]:
        ancestors = self._ancestors()
        return tuple(
            (first, second)
            for first, second in combinations(self._tasks, 2)
            if second not in ancestors[first] and first not in ancestors[second]
        )


def build_instance(process: str, n_cases: int, capacities: dict[str, int]) -> ProblemInstance:
    return ProblemInstance(build_process(process), n_cases, capacities)
