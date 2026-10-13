"""Unit tests for scenario sampling."""

import pytest

from saa_scheduling.basics.instances import JUNIOR, MODE_JUNIOR, MODE_SENIOR, SENIOR, build_instance
from saa_scheduling.distributions.duration_profiles import build_duration_model
from saa_scheduling.sampling.exceptions import (
    InvalidScenarioCountError,
    UnknownScenarioTaskError,
)
from saa_scheduling.sampling.scenario_generator import ScenarioGenerator

N_SCENARIOS = 50
CAPACITIES = {JUNIOR: 2, SENIOR: 1}


def build_generator(level: str = "D2", seed: int = 1, cases: int = 2) -> ScenarioGenerator:
    instance = build_instance("P1", cases, CAPACITIES)
    return ScenarioGenerator(instance, build_duration_model(level), seed=seed)


def test_generate_returns_the_requested_number_of_scenarios():
    assert len(build_generator().generate(N_SCENARIOS)) == N_SCENARIOS


def test_every_scenario_covers_every_task_and_mode():
    generator = build_generator()
    instance = build_instance("P1", 2, CAPACITIES)
    scenario = generator.generate(1)[0]

    expected = {
        (task, mode.name)
        for task in instance.tasks()
        for mode in instance.modes_of(task)
    }
    assert set(scenario.durations) == expected


def test_durations_are_positive_integers():
    for scenario in build_generator("D3").generate(N_SCENARIOS):
        for duration in scenario.durations.values():
            assert isinstance(duration, int)
            assert duration >= 1


def test_the_senior_mode_is_never_slower_than_the_junior_mode():
    instance = build_instance("P1", 2, CAPACITIES)
    generator = ScenarioGenerator(instance, build_duration_model("D2"), seed=5)

    for scenario in generator.generate(20):
        for task in instance.tasks():
            modes = {mode.name for mode in instance.modes_of(task)}
            if modes == {MODE_JUNIOR, MODE_SENIOR}:
                assert scenario.duration_of(task, MODE_SENIOR) <= scenario.duration_of(
                    task, MODE_JUNIOR
                )


def test_same_seed_reproduces_the_same_scenarios():
    first = build_generator(seed=7).generate(N_SCENARIOS)
    second = build_generator(seed=7).generate(N_SCENARIOS)
    assert first == second


def test_different_seeds_produce_different_scenarios():
    first = build_generator(seed=1).generate(N_SCENARIOS)
    second = build_generator(seed=2).generate(N_SCENARIOS)
    assert first != second


def test_deterministic_scenario_uses_the_mean_duration():
    instance = build_instance("P1", 1, CAPACITIES)
    model = build_duration_model("D3")
    scenario = ScenarioGenerator(instance, model, seed=0).deterministic_scenario()

    intake = "C0:Application Intake"
    expected = round(model.base_distribution("Application Intake").mean)
    assert scenario.duration_of(intake, MODE_JUNIOR) == expected


def test_robust_scenario_is_longer_than_the_mean_scenario():
    generator = build_generator("D3", cases=1)
    mean_scenario = generator.deterministic_scenario()
    robust_scenario = generator.deterministic_scenario(0.9)

    for key, duration in mean_scenario.durations.items():
        assert robust_scenario.durations[key] >= duration
    assert sum(robust_scenario.durations.values()) > sum(mean_scenario.durations.values())


def test_zero_scenarios_raises():
    with pytest.raises(InvalidScenarioCountError):
        build_generator().generate(0)


def test_unknown_task_raises():
    scenario = build_generator().generate(1)[0]
    with pytest.raises(UnknownScenarioTaskError):
        scenario.duration_of("C0:Nonexistent", MODE_JUNIOR)


def test_mixed_cases_share_the_mean_but_not_the_risk():
    from statistics import fmean, pstdev

    from saa_scheduling.distributions.duration_profiles import build_case_models

    instance = build_instance("P1", 2, CAPACITIES)
    generator = ScenarioGenerator(instance, build_case_models("MIX", 2), seed=11)
    scenarios = generator.generate(2000)

    safe = [s.duration_of("C0:Application Intake", MODE_JUNIOR) for s in scenarios]
    risky = [s.duration_of("C1:Application Intake", MODE_JUNIOR) for s in scenarios]

    assert fmean(safe) == pytest.approx(fmean(risky), rel=0.08)
    assert pstdev(risky) > 5 * pstdev(safe)


def test_the_mean_scenario_cannot_tell_mixed_cases_apart():
    from saa_scheduling.distributions.duration_profiles import build_case_models

    instance = build_instance("P1", 2, CAPACITIES)
    generator = ScenarioGenerator(instance, build_case_models("MIX", 2), seed=0)
    mean_scenario = generator.deterministic_scenario()
    robust_scenario = generator.deterministic_scenario(0.9)

    task = "Application Intake"
    assert mean_scenario.duration_of(f"C0:{task}", MODE_JUNIOR) == mean_scenario.duration_of(
        f"C1:{task}", MODE_JUNIOR
    )
    assert robust_scenario.duration_of(f"C1:{task}", MODE_JUNIOR) > robust_scenario.duration_of(
        f"C0:{task}", MODE_JUNIOR
    )


def test_a_wrong_number_of_case_models_raises():
    from saa_scheduling.distributions.duration_profiles import build_case_models

    instance = build_instance("P1", 3, CAPACITIES)
    with pytest.raises(ValueError):
        ScenarioGenerator(instance, build_case_models("MIX", 2), seed=0)
