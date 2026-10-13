"""Custom exceptions for the basics module (src/saa_scheduling/basics/exceptions.py)."""


class ProcessDefinitionError(Exception):
    """Base class for all errors raised while defining or querying a process."""


class DuplicateActivityError(ProcessDefinitionError):
    """Raised when a ProcessTemplate is given two activities with the same name."""


class UnknownActivityError(ProcessDefinitionError):
    """Raised when an activity name is referenced but not defined in the template."""


class UnknownModeError(ProcessDefinitionError):
    """Raised when an activity has no modes or an unknown mode is requested."""


class PrecedenceCycleError(ProcessDefinitionError):
    """Raised when the precedence relations of a ProcessTemplate form a cycle."""


class UnknownCaseTaskError(ProcessDefinitionError):
    """Raised when a task id does not belong to the case it is queried against."""


class UnknownRoleError(ProcessDefinitionError):
    """Raised when a role required by an activity has no capacity in the instance."""
