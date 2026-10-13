"""
Scenario generation (src/saa_scheduling/sampling/scenario_generator.py).

Draws N i.i.d. scenarios for a problem instance. For every task, one base
duration is drawn and then scaled to each mode, so the modes of a task stay
consistent within a scenario. All durations are rounded to positive integers.

The same seed reproduces the same scenarios, which is required for the SAA
replication procedure and for comparing formulations on identical samples.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from saa_scheduling.basics.instances import ProblemInstance
from saa_scheduling.distributions.activity_durations import ActivityDurationModel
from saa_scheduling.distributions.duration_models import round_duration
from saa_scheduling.sampling.exceptions import InvalidScenarioCountError
from saa_scheduling.sampling.scenario import Scenario


class ScenarioGenerator:
    """Samples scenarios for one instance.

    duration_model: one model for every case, or a sequence with one model per
        case (cases with different risk profiles).
    """

    def __init__(
        self,
        instance: ProblemInstance,
        duration_model: ActivityDurationModel | Sequence[ActivityDurationModel],
        seed: int,
    ):
        if isinstance(duration_model, ActivityDurationModel):
            models = tuple(duration_model for _ in instance.cases)
        else:
            models = tuple(duration_model)
        if len(models) != instance.n_cases:
            raise ValueError(
                f"{len(models)} duration models given for {instance.n_cases} cases."
            )
        for model in models:
            model.validate_against(instance.template)
        self._instance = instance
        self._models = models
        self._seed = seed

    def _distribution(self, task_id: str):
        model = self._models[self._instance.case_index(task_id)]
        return model.base_distribution(self._instance.activity_name_of(task_id))

    @property
    def seed(self) -> int:
        return self._seed

    def generate(self, n_scenarios: int) -> tuple[Scenario, ...]:
        if n_scenarios < 1:
            raise InvalidScenarioCountError(
                f"n_scenarios must be at least 1, got {n_scenarios}."
            )
        rng = random.Random(self._seed)
        return tuple(self._one_scenario(rng, index) for index in range(n_scenarios))

    def deterministic_scenario(self, probability: float | None = None) -> Scenario:
        """A single scenario built from the mean (or a quantile) of each activity.

        This is how the DET and ROB baselines are constructed: they solve the
        same model with one scenario that carries a single duration value per
        activity instead of a sample.
        """
        durations: dict[tuple[str, str], int] = {}
        for task_id in self._instance.tasks():
            distribution = self._distribution(task_id)
            base = (
                distribution.mean
                if probability is None
                else distribution.quantile(probability)
            )
            for mode in self._instance.modes_of(task_id):
                durations[(task_id, mode.name)] = round_duration(
                    base * mode.duration_factor
                )
        return Scenario(scenario_id=0, durations=durations)

    def _one_scenario(self, rng: random.Random, scenario_id: int) -> Scenario:
        durations: dict[tuple[str, str], int] = {}
        for task_id in self._instance.tasks():
            base = self._distribution(task_id).sample(rng)
            for mode in self._instance.modes_of(task_id):
                durations[(task_id, mode.name)] = round_duration(
                    base * mode.duration_factor
                )
        return Scenario(scenario_id=scenario_id, durations=durations)
