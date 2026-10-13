"""
Activity duration model (src/saa_scheduling/distributions/activity_durations.py).

Holds one base duration distribution per activity name. The distribution of a
concrete mode is the base distribution scaled by the mode's duration factor,
so the junior and the senior mode of an activity differ only in speed, never
in shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from saa_scheduling.distributions.duration_models import DurationDistribution
from saa_scheduling.distributions.exceptions import UnknownActivityDurationError

if TYPE_CHECKING:
    from saa_scheduling.basics.process import Mode, ProcessTemplate


class ActivityDurationModel:
    """Registry of one base DurationDistribution per activity name."""

    def __init__(self, distributions: dict[str, DurationDistribution], name: str = ""):
        self._distributions = dict(distributions)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def activity_names(self) -> tuple[str, ...]:
        return tuple(self._distributions)

    def base_distribution(self, activity_name: str) -> DurationDistribution:
        try:
            return self._distributions[activity_name]
        except KeyError as exc:
            raise UnknownActivityDurationError(
                f"No duration distribution registered for activity '{activity_name}'."
            ) from exc

    def distribution_for(self, activity_name: str, mode: "Mode") -> DurationDistribution:
        return self.base_distribution(activity_name).scaled(mode.duration_factor)

    def validate_against(self, template: "ProcessTemplate") -> None:
        """Raises if any activity of the template has no registered distribution."""
        missing = [
            name for name in template.activity_names if name not in self._distributions
        ]
        if missing:
            raise UnknownActivityDurationError(
                f"Template '{template.name}' has activities with no registered "
                f"duration distribution: {missing}."
            )
