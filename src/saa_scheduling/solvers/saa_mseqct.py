"""
SAA-MSEQCT formulation (src/saa_scheduling/solvers/saa_mseqct.py).

The sequence-based continuous-time model of Klein~(2025), extended to the SAA
setting. Resource feasibility is expressed through pairwise sequencing
variables instead of interval reasoning:

    y_ij = 1  activity i finishes before activity j starts
    z_ij = 1  activity i starts no later than activity j

Resource usage can only rise at a start event, so it is enough to require, for
every activity i and every pool, that the tasks running when i starts, that is
the tasks j with z_ji = 1 and y_ji = 0, together with i itself do not exceed
the capacity. Because y and z are first-stage decisions, these capacity
constraints do not have to be replicated per scenario; only the start times do.

Requires: pip install ortools
"""

from __future__ import annotations

import time

from ortools.sat.python import cp_model

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.execution.plan import Plan
from saa_scheduling.sampling.scenario import Scenario
from saa_scheduling.solvers.base import SolveResult, Solver, SolverConfig
from saa_scheduling.solvers.common import (
    add_objective,
    build_plan,
    compute_horizon,
    scaled_weight,
    unscale,
)

_FEASIBLE_STATUSES = (cp_model.OPTIMAL, cp_model.FEASIBLE)


class SaaMseqctSolver(Solver):
    """Sequence-based SAA formulation solved with CP-SAT."""

    name = "saa_mseqct"

    def __init__(self, config: SolverConfig | None = None):
        self._config = config or SolverConfig()

    def solve(
        self,
        instance: ProblemInstance,
        scenarios: tuple[Scenario, ...],
        visibility_weight: float = 0.0,
        due_dates: dict[str, int] | None = None,
    ) -> SolveResult:
        started = time.perf_counter()
        model, variables = self._build(instance, scenarios, visibility_weight, due_dates)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._config.max_seconds
        solver.parameters.num_workers = self._config.workers
        solver.parameters.random_seed = self._config.seed
        solver.parameters.log_search_progress = self._config.log
        status = solver.Solve(model)

        proto = model.Proto()
        n_variables, n_constraints = len(proto.variables), len(proto.constraints)

        if status not in _FEASIBLE_STATUSES:
            return SolveResult(
                plan=None,
                objective=None,
                bound=None,
                status=solver.StatusName(status),
                wall_seconds=time.perf_counter() - started,
                deterministic_seconds=solver.ResponseProto().deterministic_time,
                n_variables=n_variables,
                n_constraints=n_constraints,
            )

        return SolveResult(
            plan=self._extract_plan(instance, solver, variables),
            objective=unscale(solver.ObjectiveValue(), len(scenarios)),
            bound=unscale(solver.BestObjectiveBound(), len(scenarios)),
            status=solver.StatusName(status),
            wall_seconds=time.perf_counter() - started,
            deterministic_seconds=solver.ResponseProto().deterministic_time,
            n_variables=n_variables,
            n_constraints=n_constraints,
        )

    # ------------------------------------------------------------------
    def _build(
        self,
        instance: ProblemInstance,
        scenarios: tuple[Scenario, ...],
        visibility_weight: float,
        due_dates: dict[str, int] | None = None,
    ):
        model = cp_model.CpModel()
        horizon = compute_horizon(instance, scenarios)
        tasks = instance.tasks()
        weight = scaled_weight(visibility_weight)

        # ---- first stage: modes
        mode_used: dict[tuple[str, str], cp_model.IntVar] = {}
        role_used: dict[tuple[str, str], list] = {}
        for task in tasks:
            literals = []
            for mode in instance.modes_of(task):
                literal = model.NewBoolVar(f"mode_{task}_{mode.name}")
                mode_used[(task, mode.name)] = literal
                role_used.setdefault((task, mode.role), []).append(literal)
                literals.append(literal)
            model.AddExactlyOne(literals)

        def uses_role(task: str, role: str):
            """Linear expression: 1 if the chosen mode of `task` needs `role`."""
            return sum(role_used.get((task, role), []))

        # ---- first stage: planned start times
        planned = {task: model.NewIntVar(0, horizon, f"planned_{task}") for task in tasks}
        if not weight:
            for task in tasks:
                model.Add(planned[task] == 0)

        # ---- first stage: sequencing variables for the free pairs
        pairs = instance.conflicting_pairs()
        y: dict[tuple[str, str], cp_model.IntVar] = {}
        z: dict[tuple[str, str], cp_model.IntVar] = {}
        for first, second in pairs:
            for a, b in ((first, second), (second, first)):
                y[(a, b)] = model.NewBoolVar(f"y_{a}_{b}")
                z[(a, b)] = model.NewBoolVar(f"z_{a}_{b}")
                # finishing before j starts implies starting before j starts
                model.AddImplication(y[(a, b)], z[(a, b)])
            model.Add(y[(first, second)] + y[(second, first)] <= 1)
            model.Add(z[(first, second)] + z[(second, first)] >= 1)

        # ---- second stage: start and end times per scenario
        start: dict[tuple[str, int], cp_model.IntVar] = {}
        end: dict[tuple[str, int], cp_model.IntVar] = {}
        for index, scenario in enumerate(scenarios):
            for task in tasks:
                start[(task, index)] = model.NewIntVar(0, horizon, f"start_{task}_s{index}")
                end[(task, index)] = model.NewIntVar(0, horizon, f"end_{task}_s{index}")
                model.Add(
                    end[(task, index)]
                    == start[(task, index)]
                    + sum(
                        scenario.duration_of(task, mode.name) * mode_used[(task, mode.name)]
                        for mode in instance.modes_of(task)
                    )
                )
                model.Add(start[(task, index)] >= planned[task])
            for task in tasks:
                for predecessor in instance.predecessors_of(task):
                    model.Add(start[(task, index)] >= end[(predecessor, index)])

            # the sequencing commitments hold in every scenario
            for (a, b), literal in y.items():
                model.Add(start[(b, index)] >= end[(a, index)]).OnlyEnforceIf(literal)
            for (a, b), literal in z.items():
                model.Add(start[(a, index)] <= start[(b, index)]).OnlyEnforceIf(literal)

        # ---- resource feasibility, first stage only
        overlap: dict[tuple[str, str, str], cp_model.IntVar] = {}
        for role, capacity in instance.capacities.items():
            for task in tasks:
                running = []
                for other in tasks:
                    if (other, task) not in z:
                        continue  # ordered by precedence, so they never overlap
                    variable = model.NewBoolVar(f"runs_{other}_at_{task}_{role}")
                    overlap[(other, task, role)] = variable
                    # other occupies `role` while `task` starts if it started no
                    # later, has not finished, and uses that pool
                    model.Add(
                        variable
                        >= z[(other, task)] - y[(other, task)] + uses_role(other, role) - 1
                    )
                    running.append(variable)
                if running:
                    model.Add(sum(running) + uses_role(task, role) <= capacity)

        # ---- objective: makespan, deviation from the plan, case tardiness
        add_objective(
            model, instance, start, end, planned, len(scenarios), horizon, weight, due_dates
        )

        return model, {"mode_used": mode_used, "planned": planned, "start": start}

    # ------------------------------------------------------------------
    def _extract_plan(self, instance: ProblemInstance, solver, variables) -> Plan:
        modes = {
            task: next(
                mode.name
                for mode in instance.modes_of(task)
                if solver.Value(variables["mode_used"][(task, mode.name)])
            )
            for task in instance.tasks()
        }
        planned_starts = {
            task: int(solver.Value(variables["planned"][task])) for task in instance.tasks()
        }
        reference = {
            task: int(solver.Value(variables["start"][(task, 0)])) for task in instance.tasks()
        }
        return build_plan(instance, modes, planned_starts, reference)
