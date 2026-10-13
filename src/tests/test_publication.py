"""Unit tests for published start times and the visibility metrics."""

import pytest

from saa_scheduling.basics.instances import JUNIOR, SENIOR, build_instance
from saa_scheduling.distributions.duration_profiles import build_duration_model
from saa_scheduling.execution.plan import Plan
from saa_scheduling.execution.publication import (
    empirical_quantile,
    publish_quantile_times,
    without_start_times,
)
from saa_scheduling.execution.simulator import VISIBILITY_TOLERANCES, evaluate
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator

CAPACITIES = {JUNIOR: 2, SENIOR: 1}


def setup(level="D3", n=40, seed=3):
    instance = build_instance("P1", 2, CAPACITIES)
    generator = ScenarioGenerator(instance, build_duration_model(level), seed=seed)
    order = list(instance.tasks())
    plan = Plan.create(
        {task: instance.modes_of(task)[0].name for task in order},
        order,
        {task: 0 for task in order},
    )
    return instance, plan, generator.generate(n), generator


def test_empirical_quantile():
    values = [5, 1, 4, 2, 3]
    assert empirical_quantile(values, 0.2) == 1
    assert empirical_quantile(values, 0.5) == 3
    assert empirical_quantile(values, 1.0) == 5
    with pytest.raises(ValueError):
        empirical_quantile([], 0.5)


def test_published_times_keep_modes_and_order():
    instance, plan, scenarios, _ = setup()
    published = publish_quantile_times(instance, plan, scenarios, 0.9)

    assert dict(published.modes) == dict(plan.modes)
    assert published.order == plan.order


def test_higher_quantiles_publish_later_times():
    instance, plan, scenarios, _ = setup()
    early = publish_quantile_times(instance, plan, scenarios, 0.5)
    late = publish_quantile_times(instance, plan, scenarios, 0.95)

    for task in instance.tasks():
        assert late.planned_start_of(task) >= early.planned_start_of(task)


def test_higher_quantiles_trade_makespan_for_visibility():
    instance, plan, scenarios, generator = setup()
    holdout = generator.__class__(instance, build_duration_model("D3"), seed=99).generate(300)

    early = evaluate(instance, publish_quantile_times(instance, plan, scenarios, 0.5), holdout)
    late = evaluate(instance, publish_quantile_times(instance, plan, scenarios, 0.95), holdout)

    assert late.on_time_share_at[0] >= early.on_time_share_at[0]
    assert late.mean_makespan >= early.mean_makespan


def test_without_start_times_resets_every_promise():
    instance, plan, scenarios, _ = setup()
    published = publish_quantile_times(instance, plan, scenarios, 0.9)
    cleared = without_start_times(published)

    assert set(cleared.planned_starts.values()) == {0}
    assert cleared.order == published.order


def test_tolerant_metrics_are_never_stricter():
    instance, plan, scenarios, _ = setup()
    published = publish_quantile_times(instance, plan, scenarios, 0.8)
    result = evaluate(instance, published, scenarios)

    tolerances = sorted(VISIBILITY_TOLERANCES)
    for smaller, larger in zip(tolerances, tolerances[1:]):
        assert result.visibility_at[larger] >= result.visibility_at[smaller]
        assert result.on_time_share_at[larger] >= result.on_time_share_at[smaller]
    assert result.visibility_at[0] == result.visibility


def test_on_time_share_is_at_least_the_strict_visibility():
    instance, plan, scenarios, _ = setup()
    published = publish_quantile_times(instance, plan, scenarios, 0.8)
    result = evaluate(instance, published, scenarios)

    assert result.on_time_share_at[0] >= result.visibility_at[0]
    assert 0.0 <= result.on_time_share_at[0] <= 1.0


@pytest.mark.parametrize("probability", [0.5, 0.8, 0.9])
def test_every_promise_holds_in_at_least_q_of_the_planning_scenarios(probability):
    """The published times are self-consistent on the scenarios they came from."""
    instance, plan, scenarios, _ = setup()
    published = publish_quantile_times(instance, plan, scenarios, probability)
    result = evaluate(instance, published, scenarios)

    for task, share in result.punctuality.items():
        assert share >= probability - 1e-9, task
