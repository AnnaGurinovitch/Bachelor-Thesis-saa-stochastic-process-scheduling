"""Unit tests for the experiment definitions and the results file."""

import pytest

from evaluation.experiments import (
    EXPERIMENTS,
    LAMBDA_GRID,
    METHOD_CP,
    METHOD_DET,
    METHOD_MSEQCT,
    METHOD_ROB,
    SCENARIO_GRID,
    build_specs,
)
from evaluation.results import ResultsFile, RunRow


def make_row(seed: int = 0, method: str = METHOD_CP) -> RunRow:
    return RunRow(
        experiment="test", process="P1", cases=2, capacities="Junior2-Senior1",
        level="D2", method=method, n_scenarios=10, visibility_weight=0.0, seed=seed,
        status="OPTIMAL", objective_in_sample=100.0, bound=100.0, gap=0.0,
        wall_seconds=1.0, deterministic_seconds=0.5, n_variables=10, n_constraints=20,
        oos_makespan=110.0, oos_makespan_std=5.0, oos_visibility=0.5,
        oos_mean_delay=2.0, oos_max_delay=7,
    )


def test_every_experiment_builds():
    for name in EXPERIMENTS:
        assert build_specs(name)


def test_unknown_experiment_raises():
    with pytest.raises(ValueError):
        build_specs("E9")


def test_e1_covers_the_scenario_grid_and_the_baselines():
    specs = build_specs("E1")
    saa = [s for s in specs if s.method == METHOD_CP]

    assert {s.n_scenarios for s in saa} == set(SCENARIO_GRID)
    assert {s.level for s in specs} == {"D1", "D2", "D3"}
    # each baseline once with its own start times and once without
    assert len([s for s in specs if s.method == METHOD_DET]) == 6
    assert len([s for s in specs if s.method == METHOD_ROB]) == 6
    assert len(saa) == 75


def test_e2_runs_both_formulations_from_small_to_large():
    specs = build_specs("E2")

    assert {s.method for s in specs} == {METHOD_CP, METHOD_MSEQCT}
    assert [s.cases for s in specs] == sorted(s.cases for s in specs)
    assert len(specs) == 288


def test_e3_covers_the_lambda_grid_and_the_quantiles():
    specs = build_specs("E3")
    saa = [s for s in specs if s.method == METHOD_CP]

    assert {s.visibility_weight for s in saa} == set(LAMBDA_GRID)
    assert {s.publish for s in saa if s.visibility_weight == 0} == {
        "none", "q50", "q70", "q80", "q90", "q95"
    }


def test_e4_covers_the_complexity_grid_with_three_seeds():
    specs = build_specs("E4")
    saa = [s for s in specs if s.method == METHOD_CP]

    assert {s.cases for s in specs} == {1, 2, 3, 4, 5, 6}
    assert {s.process for s in specs if s.cases > 4} == {"P1"}
    assert {s.seed for s in saa} == {0, 1, 2}
    assert len({s.solve_key for s in saa}) == 126 * 3
    assert [s.cases for s in specs] == sorted(s.cases for s in specs)


def test_e5_varies_only_the_time_limit_within_a_configuration():
    specs = build_specs("E5")

    assert {s.max_seconds for s in specs} == {60.0, 300.0, 1200.0}
    assert len({s.solve_key for s in specs}) == len(specs) == 24
    assert len({s.label for s in specs}) == 24


def test_e3b_repeats_e3_under_its_own_name():
    e3, e3b = build_specs("E3"), build_specs("E3b")
    assert len(e3) == len(e3b)
    assert {s.experiment for s in e3b} == {"E3b"}


def test_rows_of_one_solve_are_adjacent():
    """The runner caches only the latest solve, so its rows must follow each other."""
    for name in EXPERIMENTS:
        seen, previous = set(), None
        for spec in build_specs(name):
            if spec.solve_key != previous:
                assert spec.solve_key not in seen, (name, spec.label)
                seen.add(spec.solve_key)
                previous = spec.solve_key


def test_specs_have_unique_labels():
    for name in EXPERIMENTS:
        specs = build_specs(name)
        assert len({spec.label for spec in specs}) == len(specs)


def test_results_file_round_trip(tmp_path):
    path = tmp_path / "runs.csv"
    results = ResultsFile(str(path))
    row = make_row()

    results.append(row)
    reloaded = ResultsFile(str(path))

    assert row.key in reloaded
    assert len(reloaded.rows()) == 1


def test_results_file_resumes_only_finished_runs(tmp_path):
    path = tmp_path / "runs.csv"
    results = ResultsFile(str(path))
    results.append(make_row(seed=0))
    reloaded = ResultsFile(str(path))

    assert make_row(seed=0).key in reloaded
    assert make_row(seed=1).key not in reloaded
    assert make_row(seed=0, method=METHOD_MSEQCT).key not in reloaded


def test_an_old_file_is_migrated_and_keeps_its_keys(tmp_path):
    import csv

    path = tmp_path / "old.csv"
    old_fields = [
        "experiment", "process", "cases", "capacities", "level", "method", "n_scenarios",
        "visibility_weight", "seed", "status",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(dict(experiment="test", process="P1", cases=2,
                             capacities="Junior2-Senior1", level="D2", method="det",
                             n_scenarios=1, visibility_weight=0.0, seed=0, status="OPTIMAL"))

    results = ResultsFile(str(path))
    rows = results.rows()

    assert "publish" in rows[0] and rows[0]["publish"] == "own"
    assert len(rows) == 1


def test_appending_twice_keeps_both_rows(tmp_path):
    path = tmp_path / "runs.csv"
    results = ResultsFile(str(path))
    results.append(make_row(seed=0))
    results.append(make_row(seed=1))

    assert len(results.rows()) == 2


def test_e6_contrasts_homogeneous_and_mixed_risk_under_three_objectives():
    specs = build_specs("E6")

    assert {s.level for s in specs} == {"D3", "MIX"}
    assert {(s.objective, s.sla_factor) for s in specs} == {
        ("makespan", None), ("sla", 0.85), ("sla", 1.0)
    }
    assert {s.method for s in specs} == {METHOD_CP, METHOD_DET, METHOD_ROB}
    assert [s.cases for s in specs] == sorted(s.cases for s in specs)


def test_due_dates_do_not_depend_on_the_risk_profile():
    from evaluation.due_dates import common_due_dates
    from saa_scheduling.basics.instances import build_instance
    from saa_scheduling.distributions.duration_profiles import build_case_models

    instance = build_instance("P1", 3, {"Junior": 1, "Senior": 1})
    mixed = common_due_dates(instance, build_case_models("MIX", 3), 1.0)
    risky = common_due_dates(instance, build_case_models("D3", 3), 1.0)

    assert mixed == risky
    assert len(set(mixed.values())) == 1


def test_the_holdout_sample_is_reproducible_and_independent_of_the_solve_seeds():
    from evaluation.holdout import HOLDOUT_SEED, holdout_scenarios
    from saa_scheduling.basics.instances import build_instance
    from saa_scheduling.distributions.duration_profiles import build_case_models
    from saa_scheduling.sampling.scenario_generator import ScenarioGenerator

    capacities = {"Junior": 2, "Senior": 1}
    first = holdout_scenarios("P1", 2, capacities, "D2", size=5)
    again = holdout_scenarios("P1", 2, capacities, "D2", size=5)
    assert [s.durations for s in first] == [s.durations for s in again]

    instance = build_instance("P1", 2, capacities)
    models = build_case_models("D2", 2)
    for seed in (0, 1, 2):
        assert seed != HOLDOUT_SEED
        planning = ScenarioGenerator(instance, models, seed=seed).generate(5)
        assert [s.durations for s in planning] != [s.durations for s in first]
