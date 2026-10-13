"""
Variability levels (src/saa_scheduling/distributions/duration_profiles.py).

Three duration models are defined for the same activities. They share the
mean duration of every activity and differ only in the shape of the
distributions:

  D1  near-deterministic  every activity log-normal with a small cv
  D2  realistic           one family per activity, chosen by its nature
  D3  strongly bimodal    two far apart regimes for the activities before the
                          AND-join and on the senior pool

Because the means are identical, a scheduler that only uses mean durations
receives exactly the same instance in all three levels. Differences in its
realised performance are therefore caused by the shape of the distributions.
"""

from __future__ import annotations

from saa_scheduling.distributions.activity_durations import ActivityDurationModel
from saa_scheduling.distributions.duration_models import (
    LogNormalDuration,
    UniformDuration,
    two_regime_mixture,
)

#: Mean duration of every activity in minutes; identical in all levels.
ACTIVITY_MEANS = {
    "Application Intake": 33.0,
    "Risk Assessment": 45.0,
    "Document Check": 20.0,
    "Verify Income": 15.0,
    "Verify Identity": 10.0,
    "Credit Scoring": 20.0,
    "Loan Decision": 30.0,
}

LOW_CV = 0.1
MIXTURE_COMPONENT_CV = 0.15


def _d1() -> dict:
    return {name: LogNormalDuration(mean, LOW_CV) for name, mean in ACTIVITY_MEANS.items()}


def _d2() -> dict:
    return {
        # digital submission (fast) versus paper submission (slow)
        "Application Intake": two_regime_mixture(15.0, 60.0, 0.6, MIXTURE_COMPONENT_CV),
        "Risk Assessment": LogNormalDuration(45.0, 0.22),
        "Document Check": UniformDuration(10.0, 30.0),
        "Verify Income": UniformDuration(10.0, 20.0),
        "Verify Identity": LogNormalDuration(10.0, 0.3),
        "Credit Scoring": LogNormalDuration(20.0, 0.25),
        "Loan Decision": LogNormalDuration(30.0, 0.27),
    }


def _d3() -> dict:
    return {
        "Application Intake": two_regime_mixture(15.0, 105.0, 0.8, MIXTURE_COMPONENT_CV),
        "Risk Assessment": two_regime_mixture(25.0, 105.0, 0.75, MIXTURE_COMPONENT_CV),
        "Document Check": two_regime_mixture(10.0, 60.0, 0.8, MIXTURE_COMPONENT_CV),
        "Verify Income": LogNormalDuration(15.0, 0.3),
        "Verify Identity": LogNormalDuration(10.0, 0.3),
        "Credit Scoring": LogNormalDuration(20.0, 0.3),
        "Loan Decision": LogNormalDuration(30.0, 0.3),
    }


DURATION_LEVELS = {"D1": _d1, "D2": _d2, "D3": _d3}

#: Mixed-risk case profiles: the cases of an instance alternate between the
#: listed levels. Because all levels share the same means, a mean-based planner
#: sees identical cases, while the durations of every second case are risky.
CASE_MIXES = {"MIX": ("D1", "D3")}


def build_case_models(level: str, n_cases: int) -> tuple[ActivityDurationModel, ...]:
    """One duration model per case: the same level for all, or a mix."""
    if level in CASE_MIXES:
        pattern = CASE_MIXES[level]
        return tuple(build_duration_model(pattern[i % len(pattern)]) for i in range(n_cases))
    model = build_duration_model(level)
    return tuple(model for _ in range(n_cases))


def build_duration_model(level: str) -> ActivityDurationModel:
    try:
        distributions = DURATION_LEVELS[level]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown variability level '{level}'; available: {sorted(DURATION_LEVELS)}."
        ) from exc
    return ActivityDurationModel(distributions, name=level)
