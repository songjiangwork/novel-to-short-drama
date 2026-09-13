from __future__ import annotations


class ArtifactError(Exception):
    """Base class for canonical artifact failures."""


class CanonicalSerializationError(ArtifactError, ValueError):
    """Raised when a value cannot be represented by the canonical JSON contract."""


class ArtifactValidationError(ArtifactError, ValueError):
    """Raised when an artifact reference or envelope is malformed."""


class ArtifactStoreError(ArtifactError):
    """Base class for artifact-store failures."""


class ArtifactNotFoundError(ArtifactStoreError, FileNotFoundError):
    """Raised when an exact artifact revision cannot be found."""


class ArtifactConflictError(ArtifactStoreError):
    """Raised when an immutable artifact revision would be overwritten."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when persisted artifact bytes fail integrity verification."""


class ArtifactPathError(ArtifactStoreError, ValueError):
    """Raised when an artifact identity is unsafe for filesystem storage."""
