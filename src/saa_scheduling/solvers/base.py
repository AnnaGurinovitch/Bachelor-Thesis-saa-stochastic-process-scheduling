"""
Solver interface and shared result type (src/saa_scheduling/solvers/base.py).

Every formulation returns the same SolveResult, so the experiment runner never
needs to know which formulation produced a plan. The plan itself is judged by
the simulator, not by the objective value reported here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.execution.plan import Plan
from saa_scheduling.sampling.scenario import Scenario

#: CP-SAT optimises integers, so the objective is scaled by this factor before
#: rounding. It bounds the resolution of the visibility weight lambda.
OBJECTIVE_SCALE = 100

#: With due dates, one minute of case tardiness counts this many times as much
#: as one minute of makespan. The makespan term only breaks ties between plans
#: that are equally punctual.
TARDINESS_WEIGHT = 10


@dataclass(frozen=True)
class SolverConfig:
    """Solver settings shared by both formulations."""

    max_seconds: float = 300.0
    workers: int = 4
    seed: int = 0
    log: bool = False


@dataclass(frozen=True)
class SolveResult:
    """Outcome of one solve.

    plan: the first-stage decisions, or None if no solution was found.
    objective: the in-sample SAA objective reported by the solver.
    bound: the best bound at termination; with `objective` it gives the gap.
    status: solver status, e.g. "OPTIMAL", "FEASIBLE", "UNKNOWN".
    wall_seconds: wall-clock solve time.
    deterministic_seconds: the solver's hardware-independent work counter.
    n_variables, n_constraints: size of the model that was handed to CP-SAT.
    """

    plan: Plan | None
    objective: float | None
    bound: float | None
    status: str
    wall_seconds: float
    deterministic_seconds: float
    n_variables: int
    n_constraints: int

    @property
    def gap(self) -> float | None:
        """Relative gap between incumbent and bound, or None if unavailable."""
        if self.objective is None or self.bound is None or self.objective == 0:
            return None
        return abs(self.objective - self.bound) / abs(self.objective)

    @property
    def solved(self) -> bool:
        return self.plan is not None


class Solver(ABC):
    """Common interface of the two SAA formulations."""

    name: str = "solver"

    @abstractmethod
    def solve(
        self,
        instance: ProblemInstance,
        scenarios: tuple[Scenario, ...],
        visibility_weight: float = 0.0,
        due_dates: dict[str, int] | None = None,
    ) -> SolveResult:
        """Builds and solves the SAA model for the given scenarios.

        due_dates: optional due date (SLA) per case id. With due dates the model
            minimises the expected case tardiness first and the makespan second.
        """
