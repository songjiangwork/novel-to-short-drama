from __future__ import annotations


class FoundationError(Exception):
    """Base class for v1.2 Foundation control-plane failures."""


class LineageModelError(FoundationError, ValueError):
    """Raised when a LineageRef is structurally malformed."""


class LineageResolutionError(FoundationError):
    """Raised when an exact lineage reference cannot be resolved."""


class PointerError(FoundationError):
    """Base class for current-pointer failures."""


class PointerNotFoundError(PointerError, FileNotFoundError):
    """Raised when a current pointer head does not exist."""


class PointerConflictError(PointerError):
    """Raised when a pointer compare-and-swap observes an unexpected current head."""


class PointerLockedError(PointerError):
    """Raised when a pointer update lock already exists."""


class PointerIntegrityError(PointerError):
    """Raised when persisted pointer state fails structural or integrity checks."""


class SupersessionError(FoundationError):
    """Raised when a supersession relation is malformed or cannot be verified."""


class ValidationModelError(FoundationError, ValueError):
    """Raised when structured validation data violates the Foundation contract."""


class ApprovalError(FoundationError):
    """Base class for formal semantic-approval failures."""


class ApprovalBlockedError(ApprovalError):
    """Raised when a requested APPROVE decision fails its formal gate."""


class ApprovalResolutionError(ApprovalError):
    """Raised when an exact ApprovalRef cannot be resolved and verified."""
