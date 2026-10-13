"""Custom exceptions for the solvers module."""


class SolverError(Exception):
    """Base class for all errors raised while building or solving a model."""


class ModelBuildError(SolverError):
    """Raised when an instance cannot be translated into a solver model."""


class NoSolutionError(SolverError):
    """Raised when the solver proves infeasibility."""
