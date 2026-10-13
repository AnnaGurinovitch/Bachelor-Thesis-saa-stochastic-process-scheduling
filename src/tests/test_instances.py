"""Unit tests for the process models and problem instances."""

import pytest

from saa_scheduling.basics.exceptions import (
    PrecedenceCycleError,
    UnknownCaseTaskError,
    UnknownModeError,
    UnknownRoleError,
)
from saa_scheduling.basics.instances import (
    JUNIOR,
    MODE_JUNIOR,
    MODE_SENIOR,
    SENIOR,
    SENIOR_SPEEDUP,
    build_instance,
    build_process,
)
from saa_scheduling.basics.process import Activity, Mode, ProcessTemplate

CAPACITIES = {JUNIOR: 2, SENIOR: 1}

# activities per case, free pairs with one case, free pairs with four cases
EXPECTED_STRUCTURE = {"P1": (5, 1, 154), "P2": (5, 3, 162), "P3": (7, 4, 310)}


@pytest.mark.parametrize("process", sorted(EXPECTED_STRUCTURE))
def test_structure_matches_the_documented_numbers(process):
    activities, pairs_one_case, pairs_four_cases = EXPECTED_STRUCTURE[process]
    one = build_instance(process, 1, CAPACITIES)
    four = build_instance(process, 4, CAPACITIES)

    assert len(one.tasks()) == activities
    assert len(four.tasks()) == 4 * activities
    assert len(one.free_pairs()) == pairs_one_case
    assert len(four.free_pairs()) == pairs_four_cases


@pytest.mark.parametrize("process", sorted(EXPECTED_STRUCTURE))
def test_free_pairs_are_never_precedence_related(process):
    instance = build_instance(process, 2, CAPACITIES)
    for first, second in instance.free_pairs():
        assert second not in instance.predecessors_of(first)
        assert first not in instance.predecessors_of(second)


def test_tasks_are_listed_in_precedence_order():
    instance = build_instance("P3", 2, CAPACITIES)
    position = {task: index for index, task in enumerate(instance.tasks())}
    for task in instance.tasks():
        for predecessor in instance.predecessors_of(task):
            assert position[predecessor] < position[task]


def test_junior_tasks_have_a_faster_senior_mode():
    instance = build_instance("P1", 1, CAPACITIES)
    intake = [t for t in instance.tasks() if t.endswith("Application Intake")][0]
    modes = {mode.name: mode for mode in instance.modes_of(intake)}

    assert set(modes) == {MODE_JUNIOR, MODE_SENIOR}
    assert modes[MODE_JUNIOR].duration_factor == 1.0
    assert modes[MODE_SENIOR].duration_factor == pytest.approx(SENIOR_SPEEDUP)
    assert modes[MODE_SENIOR].role == SENIOR


def test_decision_tasks_are_senior_only():
    instance = build_instance("P1", 1, CAPACITIES)
    decision = [t for t in instance.tasks() if t.endswith("Loan Decision")][0]
    modes = instance.modes_of(decision)

    assert len(modes) == 1
    assert modes[0].role == SENIOR
    assert modes[0].duration_factor == 1.0


def test_cases_do_not_share_task_ids():
    instance = build_instance("P1", 3, CAPACITIES)
    assert len(set(instance.tasks())) == len(instance.tasks())
    assert instance.case_of("C1:Risk Assessment").case_id == "C1"


def test_missing_capacity_raises():
    with pytest.raises(UnknownRoleError):
        build_instance("P1", 1, {JUNIOR: 1})


def test_zero_capacity_raises():
    with pytest.raises(UnknownRoleError):
        build_instance("P1", 1, {JUNIOR: 0, SENIOR: 1})


def test_unknown_mode_raises():
    instance = build_instance("P1", 1, CAPACITIES)
    with pytest.raises(UnknownModeError):
        instance.mode_of("C0:Loan Decision", MODE_JUNIOR)


def test_cross_case_task_id_raises():
    instance = build_instance("P1", 2, CAPACITIES)
    with pytest.raises(UnknownCaseTaskError):
        instance.cases[1].activity_name("C0:Application Intake")


def test_precedence_cycle_raises():
    mode = (Mode("only", JUNIOR),)
    with pytest.raises(PrecedenceCycleError):
        ProcessTemplate("cycle", [Activity("A", mode, ("B",)), Activity("B", mode, ("A",))])


def test_unknown_process_name_raises():
    with pytest.raises(ValueError):
        build_process("P9")


@pytest.mark.parametrize("process", sorted(EXPECTED_STRUCTURE))
def test_every_case_ends_with_its_loan_decision(process):
    instance = build_instance(process, 3, CAPACITIES)
    for case in instance.cases:
        assert instance.final_tasks(case) == (f"{case.case_id}:Loan Decision",)
