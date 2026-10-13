"""Custom exceptions for the execution module."""


class ExecutionError(Exception):
    """Base class for all errors raised while validating or executing a plan."""


class InfeasiblePlanError(ExecutionError):
    """Raised when a plan does not match its instance or violates precedence."""
