"""Unit tests for the SAA-MSEQCT formulation and its agreement with SAA-CP."""

import pytest

pytest.importorskip("ortools")

from saa_scheduling.basics.instances import JUNIOR, SENIOR, build_instance
from saa_scheduling.distributions.duration_profiles import build_duration_model
from saa_scheduling.execution.simulator import evaluate, execute
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator
from saa_scheduling.solvers.base import SolverConfig
from saa_scheduling.solvers.saa_cp import SaaCpSolver
from saa_scheduling.solvers.saa_mseqct import SaaMseqctSolver

TOLERANCE = 1e-6
BUDGET = SolverConfig(max_seconds=60.0, workers=4, seed=0)


def setup(process="P1", cases=2, capacities=None, level="D2", n=5, seed=1):
    instance = build_instance(process, cases, capacities or {JUNIOR: 2, SENIOR: 1})
    scenarios = ScenarioGenerator(instance, build_duration_model(level), seed=seed).generate(n)
    return instance, scenarios


def test_solver_returns_a_valid_plan():
    instance, scenarios = setup()
    result = SaaMseqctSolver(BUDGET).solve(instance, scenarios)

    assert result.solved
    result.plan.validate(instance)
    assert result.bound is not None


def test_the_reported_objective_matches_the_simulator():
    instance, scenarios = setup()
    result = SaaMseqctSolver(BUDGET).solve(instance, scenarios)

    realised = evaluate(instance, result.plan, scenarios)

    assert realised.objective <= result.objective + TOLERANCE


def test_the_plan_never_overloads_a_resource_pool():
    """Counts, for every scenario and every minute, how many units are busy."""
    instance, scenarios = setup(capacities={JUNIOR: 2, SENIOR: 1}, n=4)
    result = SaaMseqctSolver(BUDGET).solve(instance, scenarios)

    for scenario in scenarios:
        execution = execute(instance, result.plan, scenario)
        events = {}
        for task in instance.tasks():
            role = instance.mode_of(task, result.plan.mode_of(task)).role
            events.setdefault(role, []).append(
                (execution.starts[task], execution.ends[task])
            )
        for role, intervals in events.items():
            for time_point in {start for start, _ in intervals}:
                busy = sum(1 for start, end in intervals if start <= time_point < end)
                assert busy <= instance.capacity_of(role)


@pytest.mark.parametrize(
    "process,cases,capacities",
    [
        ("P1", 1, {JUNIOR: 1, SENIOR: 1}),
        ("P1", 2, {JUNIOR: 2, SENIOR: 1}),
        ("P2", 1, {JUNIOR: 2, SENIOR: 2}),
    ],
)
def test_the_two_formulations_agree_on_small_instances(process, cases, capacities):
    """SAA-MSEQCT commits the overlap structure as well, so it can never be
    better than SAA-CP; on small instances both should reach the same value."""
    instance, scenarios = setup(process=process, cases=cases, capacities=capacities, n=4)

    cp_result = SaaCpSolver(BUDGET).solve(instance, scenarios)
    mseqct_result = SaaMseqctSolver(BUDGET).solve(instance, scenarios)

    assert cp_result.status == "OPTIMAL"
    assert mseqct_result.status == "OPTIMAL"
    assert cp_result.objective <= mseqct_result.objective + TOLERANCE
    assert mseqct_result.objective == pytest.approx(cp_result.objective, rel=0.02)


def test_capacity_constraints_do_not_grow_with_the_scenario_count():
    """The sequencing variables are first stage, so only start times multiply."""
    instance, few = setup(n=2)
    _, many = setup(n=8)

    small = SaaMseqctSolver(BUDGET).solve(instance, few)
    large = SaaMseqctSolver(BUDGET).solve(instance, many)

    growth = large.n_constraints / small.n_constraints
    assert growth < 4.0


def test_a_visibility_weight_reduces_the_delay():
    instance, scenarios = setup(level="D3", n=5)
    solver = SaaMseqctSolver(BUDGET)

    plain = evaluate(instance, solver.solve(instance, scenarios, 0.0).plan, scenarios)
    visible = evaluate(instance, solver.solve(instance, scenarios, 2.0).plan, scenarios)

    assert visible.mean_total_delay <= plain.mean_total_delay + TOLERANCE
