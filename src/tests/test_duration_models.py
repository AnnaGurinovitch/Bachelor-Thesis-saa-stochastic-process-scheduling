"""Unit tests for the duration distributions and the three variability levels."""

import random

import pytest

from saa_scheduling.distributions.duration_models import (
    LogNormalDuration,
    MixtureDuration,
    UniformDuration,
    round_duration,
    two_regime_mixture,
)
from saa_scheduling.distributions.duration_profiles import (
    ACTIVITY_MEANS,
    DURATION_LEVELS,
    build_duration_model,
)
from saa_scheduling.distributions.exceptions import (
    InvalidDistributionError,
    UnknownActivityDurationError,
)

SAMPLE_COUNT = 20000
RELATIVE_TOLERANCE = 0.05


@pytest.mark.parametrize(
    "distribution",
    [
        LogNormalDuration(45.0, 0.5),
        UniformDuration(10.0, 30.0),
        two_regime_mixture(15.0, 105.0, 0.8, 0.15),
    ],
)
def test_empirical_mean_matches_declared_mean(distribution):
    rng = random.Random(7)
    samples = [distribution.sample(rng) for _ in range(SAMPLE_COUNT)]
    assert sum(samples) / SAMPLE_COUNT == pytest.approx(
        distribution.mean, rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "distribution",
    [
        LogNormalDuration(45.0, 0.5),
        UniformDuration(10.0, 30.0),
        two_regime_mixture(15.0, 105.0, 0.8, 0.15),
    ],
)
def test_quantile_inverts_the_cdf(distribution):
    for probability in (0.1, 0.5, 0.9, 0.99):
        value = distribution.quantile(probability)
        assert distribution.cdf(value) == pytest.approx(probability, abs=1e-6)


@pytest.mark.parametrize(
    "distribution",
    [
        LogNormalDuration(45.0, 0.5),
        UniformDuration(10.0, 30.0),
        two_regime_mixture(15.0, 105.0, 0.8, 0.15),
    ],
)
def test_scaling_scales_the_mean_and_keeps_the_family(distribution):
    scaled = distribution.scaled(0.6)
    assert scaled.mean == pytest.approx(0.6 * distribution.mean)
    assert type(scaled) is type(distribution)


def test_rounding_never_returns_zero():
    assert round_duration(0.2) == 1
    assert round_duration(2.5) == 3
    assert round_duration(2.4) == 2


def test_invalid_parameters_raise():
    with pytest.raises(InvalidDistributionError):
        LogNormalDuration(10.0, 0.0)
    with pytest.raises(InvalidDistributionError):
        UniformDuration(30.0, 10.0)
    with pytest.raises(InvalidDistributionError):
        MixtureDuration((LogNormalDuration(5.0, 0.2),), (0.5,))


@pytest.mark.parametrize("level", sorted(DURATION_LEVELS))
def test_every_level_uses_the_same_means(level):
    model = build_duration_model(level)
    for activity_name, mean in ACTIVITY_MEANS.items():
        assert model.base_distribution(activity_name).mean == pytest.approx(mean)


def test_levels_differ_in_shape():
    spreads = {}
    for level in sorted(DURATION_LEVELS):
        rng = random.Random(3)
        distribution = build_duration_model(level).base_distribution("Application Intake")
        samples = [distribution.sample(rng) for _ in range(SAMPLE_COUNT)]
        mean = sum(samples) / SAMPLE_COUNT
        spreads[level] = (sum((s - mean) ** 2 for s in samples) / SAMPLE_COUNT) ** 0.5

    assert spreads["D1"] < spreads["D2"] < spreads["D3"]


def test_unknown_activity_raises():
    model = build_duration_model("D1")
    with pytest.raises(UnknownActivityDurationError):
        model.base_distribution("Nonexistent Activity")
