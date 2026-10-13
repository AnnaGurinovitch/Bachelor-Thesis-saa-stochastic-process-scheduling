"""Custom exceptions for the sampling module."""


class SamplingError(Exception):
    """Base class for all errors raised while generating or querying scenarios."""


class UnknownScenarioTaskError(SamplingError):
    """Raised when a duration is requested for a task not present in a scenario."""


class InvalidScenarioCountError(SamplingError):
    """Raised when a non-positive number of scenarios is requested."""
