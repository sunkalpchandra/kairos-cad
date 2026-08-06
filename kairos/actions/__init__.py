"""Structured CAD actions: the only interface the agent may use.

The policy emits ``Action`` objects (operation + target + typed parameters +
confidence). They are validated against per-operation parameter specs and
dispatched onto the ``CADEngine``; arbitrary code execution is impossible by
construction.
"""

from kairos.actions.executor import ActionExecutor
from kairos.actions.masking import StateFlags, legal_operations
from kairos.actions.parameters import ActionValidationError, validate_action
from kairos.actions.schema import Action, ActionResult, Operation

__all__ = [
    "Action",
    "ActionResult",
    "Operation",
    "ActionValidationError",
    "validate_action",
    "ActionExecutor",
    "StateFlags",
    "legal_operations",
]
