"""
SAA-CP formulation (src/saa_scheduling/solvers/saa_cp.py).

Expresses the SAA problem with the scheduling primitives of CP-SAT: one
optional interval per task, mode and scenario, cumulative constraints for the
resource pools, and end-before-start constraints for the precedence relations.

The first stage -- modes, committed start order and planned start times -- is
shared by all scenarios. Without the shared order, CP-SAT would be free to
sequence every scenario differently, which would amount to knowing the
durations in advance and would only yield a lower bound instead of an
executable plan.

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


class SaaCpSolver(Solver):
    """Interval-based SAA formulation solved with CP-SAT."""

    name = "saa_cp"

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
        n_scenarios = len(scenarios)

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

        plan = self._extract_plan(instance, solver, variables)
        return SolveResult(
            plan=plan,
            objective=unscale(solver.ObjectiveValue(), n_scenarios),
            bound=unscale(solver.BestObjectiveBound(), n_scenarios),
            status=solver.StatusName(status),
            wall_seconds=time.perf_counter() - started,
            deterministic_seconds=solver.ResponseProto().deterministic_time,
            n_variables=n_variables,
            n_constraints=n_constraints,
        )

    # ------------------------------------------------------------------
    # model construction
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

        # ---- first stage: modes and planned start times
        mode_used: dict[tuple[str, str], cp_model.IntVar] = {}
        for task in tasks:
            literals = []
            for mode in instance.modes_of(task):
                literal = model.NewBoolVar(f"mode_{task}_{mode.name}")
                mode_used[(task, mode.name)] = literal
                literals.append(literal)
            model.AddExactlyOne(literals)

        planned = {
            task: model.NewIntVar(0, horizon, f"planned_{task}") for task in tasks
        }
        if not weight:
            # With no visibility weight the planned times carry no meaning and
            # would only add symmetric optima, so they are fixed to zero. Such a
            # plan commits modes and order, but promises no start times.
            for task in tasks:
                model.Add(planned[task] == 0)

        # ---- second stage: one start and end per task and scenario
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
                # the plan is binding: no task starts before its planned time
                model.Add(start[(task, index)] >= planned[task])

        # ---- precedence, per scenario
        for index in range(len(scenarios)):
            for task in tasks:
                for predecessor in instance.predecessors_of(task):
                    model.Add(start[(task, index)] >= end[(predecessor, index)])

        # ---- resources: one cumulative constraint per pool and scenario
        for index, scenario in enumerate(scenarios):
            per_role: dict[str, list] = {role: [] for role in instance.capacities}
            for task in tasks:
                for mode in instance.modes_of(task):
                    interval = model.NewOptionalIntervalVar(
                        start[(task, index)],
                        scenario.duration_of(task, mode.name),
                        end[(task, index)],
                        mode_used[(task, mode.name)],
                        f"itv_{task}_{mode.name}_s{index}",
                    )
                    per_role[mode.role].append(interval)
            for role, intervals in per_role.items():
                if intervals:
                    model.AddCumulative(
                        intervals, [1] * len(intervals), instance.capacity_of(role)
                    )

        # ---- shared start order: the same commitment in every scenario,
        # for every pair of tasks that could claim the same pool
        before: dict[tuple[str, str], cp_model.IntVar] = {}
        for first, second in instance.conflicting_pairs():
            literal = model.NewBoolVar(f"before_{first}_{second}")
            before[(first, second)] = literal
            for index in range(len(scenarios)):
                model.Add(start[(first, index)] <= start[(second, index)]).OnlyEnforceIf(
                    literal
                )
                model.Add(start[(second, index)] <= start[(first, index)]).OnlyEnforceIf(
                    literal.Not()
                )

        # ---- objective: makespan, deviation from the plan, case tardiness
        add_objective(
            model, instance, start, end, planned, len(scenarios), horizon, weight, due_dates
        )

        return model, {
            "mode_used": mode_used,
            "planned": planned,
            "start": start,
        }

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
            task: int(solver.Value(variables["start"][(task, 0)]))
            for task in instance.tasks()
        }
        return build_plan(instance, modes, planned_starts, reference)
