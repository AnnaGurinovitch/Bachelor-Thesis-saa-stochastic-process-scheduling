"""Custom exceptions for the distributions module."""


class DistributionError(Exception):
    """Base class for all errors raised by duration distributions and models."""


class InvalidDistributionError(DistributionError):
    """Raised when a duration distribution is defined with inconsistent parameters."""


class UnknownActivityDurationError(DistributionError):
    """Raised when a duration is requested for an activity with no registered distribution."""
