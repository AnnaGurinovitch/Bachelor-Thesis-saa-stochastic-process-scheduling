"""
Scenario (src/saa_scheduling/sampling/scenario.py).

One sampled realisation of every task duration, for every mode the task could
be executed in. Solvers and the simulator consume scenarios and never sample
durations themselves, which keeps all randomness in the sampling module.
"""

from __future__ import annotations

from types import MappingProxyType

from saa_scheduling.sampling.exceptions import UnknownScenarioTaskError


class Scenario:
    """Realised durations of one scenario, keyed by (task id, mode name).

    The durations of the modes of one task come from a single underlying draw,
    scaled by the modes' duration factors. A senior officer is therefore faster
    on exactly the cases in which a junior analyst is slow.
    """

    __slots__ = ("_scenario_id", "_durations")

    def __init__(self, scenario_id: int, durations: dict[tuple[str, str], int]):
        self._scenario_id = scenario_id
        self._durations = MappingProxyType(dict(durations))

    @property
    def scenario_id(self) -> int:
        return self._scenario_id

    @property
    def durations(self) -> MappingProxyType:
        return self._durations

    def duration_of(self, task_id: str, mode_name: str) -> int:
        try:
            return self._durations[(task_id, mode_name)]
        except KeyError as exc:
            raise UnknownScenarioTaskError(
                f"Task '{task_id}' has no sampled duration for mode '{mode_name}' "
                f"in scenario {self._scenario_id}."
            ) from exc

    def max_duration(self, task_id: str) -> int:
        return max(
            duration
            for (task, _), duration in self._durations.items()
            if task == task_id
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Scenario)
            and self._scenario_id == other._scenario_id
            and dict(self._durations) == dict(other._durations)
        )

    def __repr__(self) -> str:
        return f"Scenario(id={self._scenario_id}, tasks={len(self._durations)})"
