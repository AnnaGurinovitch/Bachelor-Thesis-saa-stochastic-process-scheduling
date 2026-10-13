"""
Duration distributions (src/saa_scheduling/distributions/duration_models.py).

Every distribution is parametrised by its mean, so that variants of the same
activity can be compared at an identical expected duration and only the shape
changes. Each distribution can be sampled, evaluated (cdf), inverted
(quantile) and scaled by a constant factor, which is how the faster senior
mode of an activity is derived from its base distribution.

Durations are expressed in minutes and are rounded to positive integers
before they enter a scenario, because CP-SAT requires integer data.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import NormalDist

from saa_scheduling.distributions.exceptions import InvalidDistributionError

MINIMUM_INTEGER_DURATION = 1
PROBABILITY_SUM_TOLERANCE = 1e-6
_QUANTILE_TOLERANCE = 1e-9
_QUANTILE_MAX_ITERATIONS = 200


def round_duration(value: float, minimum: int = MINIMUM_INTEGER_DURATION) -> int:
    """Rounds to the nearest whole minute, never below `minimum`."""
    return max(minimum, math.floor(value + 0.5))


class DurationDistribution(ABC):
    """Interface of every activity duration model."""

    @property
    @abstractmethod
    def mean(self) -> float:
        """Expected duration in minutes."""

    @abstractmethod
    def sample(self, rng: random.Random) -> float:
        """Draws one realised duration."""

    @abstractmethod
    def cdf(self, value: float) -> float:
        """Probability of a duration not exceeding `value`."""

    @abstractmethod
    def scaled(self, factor: float) -> "DurationDistribution":
        """The distribution of `factor` times this duration."""

    def quantile(self, probability: float) -> float:
        """Smallest duration whose cdf reaches `probability` (bisection)."""
        if not 0.0 < probability < 1.0:
            raise InvalidDistributionError(
                f"quantile probability must lie in (0, 1), got {probability}."
            )
        low, high = 0.0, max(self.mean, 1.0)
        while self.cdf(high) < probability:
            high *= 2.0
        for _ in range(_QUANTILE_MAX_ITERATIONS):
            middle = 0.5 * (low + high)
            if self.cdf(middle) < probability:
                low = middle
            else:
                high = middle
            if high - low < _QUANTILE_TOLERANCE:
                break
        return 0.5 * (low + high)


@dataclass(frozen=True)
class LogNormalDuration(DurationDistribution):
    """Right-skewed, strictly positive duration, given by mean and cv."""

    mean_value: float
    cv: float

    def __post_init__(self) -> None:
        if self.mean_value <= 0:
            raise InvalidDistributionError(f"mean must be positive, got {self.mean_value}.")
        if self.cv <= 0:
            raise InvalidDistributionError(f"cv must be positive, got {self.cv}.")

    @property
    def mean(self) -> float:
        return self.mean_value

    @property
    def sigma(self) -> float:
        return math.sqrt(math.log(1.0 + self.cv ** 2))

    @property
    def mu(self) -> float:
        return math.log(self.mean_value) - 0.5 * self.sigma ** 2

    def sample(self, rng: random.Random) -> float:
        return rng.lognormvariate(self.mu, self.sigma)

    def cdf(self, value: float) -> float:
        if value <= 0:
            return 0.0
        return NormalDist().cdf((math.log(value) - self.mu) / self.sigma)

    def quantile(self, probability: float) -> float:
        return math.exp(self.mu + self.sigma * NormalDist().inv_cdf(probability))

    def scaled(self, factor: float) -> "LogNormalDuration":
        return LogNormalDuration(self.mean_value * factor, self.cv)


@dataclass(frozen=True)
class UniformDuration(DurationDistribution):
    """Bounded duration with no clustering inside its range."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower <= 0:
            raise InvalidDistributionError(f"lower bound must be positive, got {self.lower}.")
        if self.upper <= self.lower:
            raise InvalidDistributionError(
                f"upper bound ({self.upper}) must exceed lower bound ({self.lower})."
            )

    @property
    def mean(self) -> float:
        return 0.5 * (self.lower + self.upper)

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.lower, self.upper)

    def cdf(self, value: float) -> float:
        if value <= self.lower:
            return 0.0
        if value >= self.upper:
            return 1.0
        return (value - self.lower) / (self.upper - self.lower)

    def quantile(self, probability: float) -> float:
        return self.lower + probability * (self.upper - self.lower)

    def scaled(self, factor: float) -> "UniformDuration":
        return UniformDuration(self.lower * factor, self.upper * factor)


@dataclass(frozen=True)
class MixtureDuration(DurationDistribution):
    """Mixture of several duration distributions, e.g. two operating regimes.

    components: the regimes, for instance a fast digital and a slow paper one.
    weights: probability of each regime; must be non-negative and sum to one.
    """

    components: tuple[DurationDistribution, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.components) != len(self.weights):
            raise InvalidDistributionError(
                f"{len(self.components)} components but {len(self.weights)} weights."
            )
        if not self.components:
            raise InvalidDistributionError("a mixture needs at least one component.")
        if any(weight < 0 for weight in self.weights):
            raise InvalidDistributionError("weights must be non-negative.")
        total = sum(self.weights)
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise InvalidDistributionError(f"weights must sum to 1.0, got {total}.")

    @property
    def mean(self) -> float:
        return sum(w * c.mean for w, c in zip(self.weights, self.components))

    def sample(self, rng: random.Random) -> float:
        component = rng.choices(self.components, weights=self.weights, k=1)[0]
        return component.sample(rng)

    def cdf(self, value: float) -> float:
        return sum(w * c.cdf(value) for w, c in zip(self.weights, self.components))

    def scaled(self, factor: float) -> "MixtureDuration":
        return MixtureDuration(
            tuple(c.scaled(factor) for c in self.components), self.weights
        )


def two_regime_mixture(
    fast_mean: float, slow_mean: float, fast_weight: float, component_cv: float
) -> MixtureDuration:
    """A bimodal duration: a fast and a slow regime, each log-normal."""
    return MixtureDuration(
        components=(
            LogNormalDuration(fast_mean, component_cv),
            LogNormalDuration(slow_mean, component_cv),
        ),
        weights=(fast_weight, 1.0 - fast_weight),
    )
