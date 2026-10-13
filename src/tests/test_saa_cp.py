"""Unit tests for the SAA-CP formulation. Require ortools."""

import pytest

pytest.importorskip("ortools")

from saa_scheduling.basics.instances import JUNIOR, MODE_SENIOR, SENIOR, build_instance
from saa_scheduling.distributions.duration_profiles import build_duration_model
from saa_scheduling.execution.simulator import evaluate
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator
from saa_scheduling.solvers.base import SolverConfig
from saa_scheduling.solvers.saa_cp import SaaCpSolver

TOLERANCE = 1e-6
BUDGET = SolverConfig(max_seconds=30.0, workers=4, seed=0)


def setup(process="P1", cases=2, capacities=None, level="D2", n=8, seed=1):
    instance = build_instance(process, cases, capacities or {JUNIOR: 2, SENIOR: 1})
    scenarios = ScenarioGenerator(instance, build_duration_model(level), seed=seed).generate(n)
    return instance, scenarios


def test_solver_returns_a_valid_plan():
    instance, scenarios = setup()
    result = SaaCpSolver(BUDGET).solve(instance, scenarios)

    assert result.solved
    assert result.status in ("OPTIMAL", "FEASIBLE")
    result.plan.validate(instance)
    assert set(result.plan.order) == set(instance.tasks())
    assert result.n_variables > 0 and result.n_constraints > 0


def test_the_reported_objective_matches_the_simulator():
    """The plan, executed by the simulator, must not be worse than the model says.

    The model may leave a task idle where the simulator starts it as early as
    possible, so the simulator value can be lower, but never higher.
    """
    instance, scenarios = setup()
    result = SaaCpSolver(BUDGET).solve(instance, scenarios)

    realised = evaluate(instance, result.plan, scenarios, visibility_weight=0.0)

    assert realised.objective <= result.objective + TOLERANCE


def test_the_gap_is_reported():
    instance, scenarios = setup(n=5)
    result = SaaCpSolver(BUDGET).solve(instance, scenarios)

    assert result.bound is not None
    assert result.gap is not None
    assert result.gap >= -TOLERANCE
    if result.status == "OPTIMAL":
        assert result.gap < 1e-4


def test_the_plan_beats_a_naive_order_in_sample():
    instance, scenarios = setup(cases=2, n=8)
    result = SaaCpSolver(BUDGET).solve(instance, scenarios)

    from saa_scheduling.execution.plan import Plan

    naive = Plan.create(
        {task: instance.modes_of(task)[0].name for task in instance.tasks()},
        list(instance.tasks()),
        {task: 0 for task in instance.tasks()},
    )
    naive_result = evaluate(instance, naive, scenarios)
    solved_result = evaluate(instance, result.plan, scenarios)

    assert solved_result.mean_makespan <= naive_result.mean_makespan + TOLERANCE


def test_more_capacity_never_lengthens_the_optimal_makespan():
    tight_instance, scenarios = setup(capacities={JUNIOR: 1, SENIOR: 1}, n=5)
    wide_instance, _ = setup(capacities={JUNIOR: 3, SENIOR: 2}, n=5)

    tight = SaaCpSolver(BUDGET).solve(tight_instance, scenarios)
    wide = SaaCpSolver(BUDGET).solve(wide_instance, scenarios)

    if tight.status == "OPTIMAL" and wide.status == "OPTIMAL":
        assert wide.objective <= tight.objective + TOLERANCE


def test_the_plan_is_executable_in_every_scenario():
    """Non-anticipativity: one plan, executed unchanged in all scenarios."""
    instance, scenarios = setup(cases=2, n=6)
    result = SaaCpSolver(BUDGET).solve(instance, scenarios)

    from saa_scheduling.execution.simulator import execute

    for scenario in scenarios:
        execution = execute(instance, result.plan, scenario)
        assert execution.makespan > 0
        assert set(execution.starts) == set(instance.tasks())


def test_a_single_resource_unit_is_claimed_in_the_committed_order():
    """With one unit per pool, that unit serves the tasks in the plan's order."""
    instance, scenarios = setup(cases=2, capacities={JUNIOR: 1, SENIOR: 1}, n=5)
    result = SaaCpSolver(BUDGET).solve(instance, scenarios)

    from saa_scheduling.execution.simulator import execute

    for scenario in scenarios:
        execution = execute(instance, result.plan, scenario)
        for role in instance.capacities:
            starts = [
                execution.starts[task]
                for task in result.plan.order
                if instance.mode_of(task, result.plan.mode_of(task)).role == role
            ]
            assert starts == sorted(starts)


def test_a_visibility_weight_improves_visibility():
    instance, scenarios = setup(level="D3", cases=2, n=8)
    solver = SaaCpSolver(BUDGET)

    without = solver.solve(instance, scenarios, visibility_weight=0.0)
    with_weight = solver.solve(instance, scenarios, visibility_weight=2.0)

    plain = evaluate(instance, without.plan, scenarios)
    visible = evaluate(instance, with_weight.plan, scenarios)

    assert visible.mean_total_delay <= plain.mean_total_delay + TOLERANCE
    assert visible.mean_makespan >= plain.mean_makespan - TOLERANCE


def test_planned_times_are_zero_without_a_visibility_weight():
    instance, scenarios = setup(n=4)
    result = SaaCpSolver(BUDGET).solve(instance, scenarios, visibility_weight=0.0)

    assert set(result.plan.planned_starts.values()) == {0}


def test_a_senior_only_task_never_gets_a_junior_mode():
    instance, scenarios = setup(n=4)
    result = SaaCpSolver(BUDGET).solve(instance, scenarios)

    for task in instance.tasks():
        if task.endswith("Loan Decision") or task.endswith("Risk Assessment"):
            assert result.plan.mode_of(task) == MODE_SENIOR


def test_with_due_dates_the_objective_matches_the_simulator():
    from evaluation.due_dates import common_due_dates
    from saa_scheduling.distributions.duration_profiles import build_case_models
    from saa_scheduling.solvers.base import TARDINESS_WEIGHT

    instance = build_instance("P1", 2, {JUNIOR: 1, SENIOR: 1})
    models = build_case_models("MIX", 2)
    scenarios = ScenarioGenerator(instance, models, seed=2).generate(6)
    due_dates = common_due_dates(instance, models, 0.85)

    result = SaaCpSolver(BUDGET).solve(instance, scenarios, due_dates=due_dates)
    realised = evaluate(instance, result.plan, scenarios, due_dates=due_dates)
    simulated = realised.mean_makespan + TARDINESS_WEIGHT * realised.mean_total_tardiness

    assert result.solved
    assert simulated <= result.objective + TOLERANCE


def test_saa_is_at_least_as_punctual_as_det_on_its_own_scenarios():
    """In-sample, SAA should not lose to the mean-based plan."""
    from evaluation.baselines import solve_baseline
    from evaluation.due_dates import common_due_dates
    from saa_scheduling.distributions.duration_profiles import build_case_models
    from saa_scheduling.solvers.base import TARDINESS_WEIGHT

    instance = build_instance("P1", 2, {JUNIOR: 1, SENIOR: 1})
    models = build_case_models("MIX", 2)
    scenarios = ScenarioGenerator(instance, models, seed=4).generate(8)
    due_dates = common_due_dates(instance, models, 0.85)
    solver = SaaCpSolver(BUDGET)

    saa = solver.solve(instance, scenarios, due_dates=due_dates)
    det, _ = solve_baseline(instance, models, solver, None, due_dates)

    def cost(plan):
        r = evaluate(instance, plan, scenarios, due_dates=due_dates)
        return r.mean_makespan + TARDINESS_WEIGHT * r.mean_total_tardiness

    # the simulator lets tasks of different pools overtake each other, which the
    # model does not, so allow a small margin instead of a strict inequality
    if saa.status == "OPTIMAL":
        assert cost(saa.plan) <= 1.02 * cost(det.plan) + TOLERANCE
