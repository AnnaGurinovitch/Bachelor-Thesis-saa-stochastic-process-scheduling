"""Unit tests for plan execution and the metrics derived from it."""

import pytest

from saa_scheduling.basics.instances import JUNIOR, MODE_SENIOR, SENIOR, build_instance
from saa_scheduling.distributions.duration_profiles import build_duration_model
from saa_scheduling.execution.exceptions import InfeasiblePlanError
from saa_scheduling.execution.plan import Plan
from saa_scheduling.execution.simulator import evaluate, execute
from saa_scheduling.sampling.scenario import Scenario
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator

TIGHT = {JUNIOR: 1, SENIOR: 1}
WIDE = {JUNIOR: 2, SENIOR: 2}

INTAKE = "C0:Application Intake"
RISK = "C0:Risk Assessment"
CHECK = "C0:Document Check"
SCORING = "C0:Credit Scoring"
DECISION = "C0:Loan Decision"
ORDER = [INTAKE, RISK, CHECK, SCORING, DECISION]


def fixed_scenario(instance, junior_durations: dict[str, int]) -> Scenario:
    """A scenario with hand-chosen junior durations; senior modes are 60 percent."""
    durations = {}
    for task in instance.tasks():
        base = junior_durations[task]
        for mode in instance.modes_of(task):
            durations[(task, mode.name)] = max(1, round(base * mode.duration_factor))
    return Scenario(0, durations)


def all_junior(instance) -> dict[str, str]:
    return {task: instance.modes_of(task)[0].name for task in instance.tasks()}


def test_hand_computed_schedule_with_one_junior_and_one_senior():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    plan = Plan.create(all_junior(instance), ORDER, {task: 0 for task in ORDER})

    execution = execute(instance, plan, scenario)

    # Intake occupies the single junior from 0 to 10. Risk Assessment runs on the
    # senior from 10 to 30. Document Check needs the junior, free at 10, so it runs
    # 10 to 40. Credit Scoring waits for both branches: 40 to 45. Loan Decision
    # follows on the senior: 45 to 60.
    assert execution.starts[INTAKE] == 0
    assert execution.starts[RISK] == 10
    assert execution.starts[CHECK] == 10
    assert execution.starts[SCORING] == 40
    assert execution.starts[DECISION] == 45
    assert execution.makespan == 60


def test_planned_start_times_delay_a_task_that_could_start_earlier():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    planned = {task: 0 for task in ORDER}
    planned[CHECK] = 25
    plan = Plan.create(all_junior(instance), ORDER, planned)

    execution = execute(instance, plan, scenario)

    assert execution.starts[CHECK] == 25  # not 10: the plan says 25
    assert execution.makespan == 75


def test_the_senior_mode_shortens_a_junior_task():
    instance = build_instance("P1", 1, WIDE)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    modes = all_junior(instance)
    modes[CHECK] = MODE_SENIOR
    plan = Plan.create(modes, ORDER, {task: 0 for task in ORDER})

    execution = execute(instance, plan, scenario)

    assert execution.ends[CHECK] - execution.starts[CHECK] == 18  # 0.6 * 30


def test_a_plan_that_matches_its_scenario_is_fully_visible():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    plan = Plan.create(all_junior(instance), ORDER, {task: 0 for task in ORDER})
    realised = execute(instance, plan, scenario)
    honest_plan = Plan.create(all_junior(instance), ORDER, dict(realised.starts))

    result = evaluate(instance, honest_plan, (scenario,))

    assert result.visibility == 1.0
    assert result.mean_total_delay == 0.0
    assert result.objective == pytest.approx(result.mean_makespan)


def test_visibility_and_delay_react_to_longer_durations():
    instance = build_instance("P1", 1, TIGHT)
    short = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    long = fixed_scenario(
        instance, {INTAKE: 40, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    realised = execute(
        instance,
        Plan.create(all_junior(instance), ORDER, {task: 0 for task in ORDER}),
        short,
    )
    plan = Plan.create(all_junior(instance), ORDER, dict(realised.starts))

    result = evaluate(instance, plan, (short, long))

    assert result.visibility == 0.5
    assert result.mean_total_delay > 0
    assert result.punctuality[INTAKE] == 1.0  # the first task always starts on time


def test_the_visibility_weight_penalises_deviation():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 40, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    plan = Plan.create(all_junior(instance), ORDER, {task: 0 for task in ORDER})

    without = evaluate(instance, plan, (scenario,), visibility_weight=0.0)
    with_weight = evaluate(instance, plan, (scenario,), visibility_weight=1.0)

    assert with_weight.objective > without.objective
    assert with_weight.mean_makespan == without.mean_makespan


def test_more_capacity_never_lengthens_this_schedule():
    scenario_durations = {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    tight_instance = build_instance("P1", 1, TIGHT)
    wide_instance = build_instance("P1", 1, WIDE)
    plan_tight = Plan.create(
        all_junior(tight_instance), ORDER, {task: 0 for task in ORDER}
    )

    tight = execute(tight_instance, plan_tight, fixed_scenario(tight_instance, scenario_durations))
    wide = execute(wide_instance, plan_tight, fixed_scenario(wide_instance, scenario_durations))

    assert wide.makespan <= tight.makespan


def test_an_order_violating_precedence_raises():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    bad_order = list(reversed(ORDER))
    plan = Plan.create(all_junior(instance), bad_order, {task: 0 for task in ORDER})

    with pytest.raises(InfeasiblePlanError):
        execute(instance, plan, scenario)


def test_an_incomplete_plan_raises():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    plan = Plan.create(all_junior(instance), ORDER[:-1], {task: 0 for task in ORDER})

    with pytest.raises(InfeasiblePlanError):
        execute(instance, plan, scenario)


def test_sampled_scenarios_run_through_the_simulator():
    instance = build_instance("P3", 2, WIDE)
    generator = ScenarioGenerator(instance, build_duration_model("D3"), seed=4)
    scenarios = generator.generate(25)
    order = list(instance.tasks())
    plan = Plan.create(
        {task: instance.modes_of(task)[0].name for task in order},
        order,
        {task: 0 for task in order},
    )

    result = evaluate(instance, plan, scenarios)

    assert result.n_scenarios == 25
    assert result.mean_makespan > 0
    assert 0.0 <= result.visibility <= 1.0


def test_case_completion_and_tardiness_are_hand_computed():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    plan = Plan.create(all_junior(instance), ORDER, {task: 0 for task in ORDER})

    # the case completes with its loan decision at 60 (see the first test)
    assert execute(instance, plan, scenario).case_completions(instance) == {"C0": 60}

    late = evaluate(instance, plan, (scenario,), due_dates={"C0": 50})
    early = evaluate(instance, plan, (scenario,), due_dates={"C0": 70})

    assert late.mean_total_tardiness == 10
    assert late.sla_violation_rate == 1.0
    assert early.mean_total_tardiness == 0
    assert early.sla_violation_rate == 0.0
    assert late.mean_case_completion == 60


def test_without_due_dates_no_tardiness_is_reported():
    instance = build_instance("P1", 1, TIGHT)
    scenario = fixed_scenario(
        instance, {INTAKE: 10, RISK: 20, CHECK: 30, SCORING: 5, DECISION: 15}
    )
    plan = Plan.create(all_junior(instance), ORDER, {task: 0 for task in ORDER})
    result = evaluate(instance, plan, (scenario,))

    assert result.mean_total_tardiness is None
    assert result.sla_violation_rate is None
