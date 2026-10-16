"""G1 shared immutable user-direction and revision-request primitives."""

from .models import (
    GUIDANCE_SCHEMA_VERSION, REVISION_REQUEST_ARTIFACT_TYPE, USER_DIRECTION_ARTIFACT_TYPE,
    USER_DIRECTION_SET_ARTIFACT_TYPE, DirectionCategory, DirectionConflictError, DirectionOrigin,
    DirectionStage, DirectionStatus, DirectionStrength, GuidanceError, RevisionRequest, TargetRef,
    TargetScope, UserDirection, UserDirectionSet,
)
from .persistence import (
    load_revision_request, load_user_direction_set, persist_revision_request,
    persist_user_direction, persist_user_direction_set, validate_revision_request,
)
from .projection import (
    EMPTY_EFFECTIVE_USER_DIRECTION_SET_HASH, EffectiveUserDirectionSet, TargetCatalog,
    project_effective_directions, resolve_direction_set, validate_direction_targets,
)

__all__ = [
    "GUIDANCE_SCHEMA_VERSION", "REVISION_REQUEST_ARTIFACT_TYPE", "USER_DIRECTION_ARTIFACT_TYPE",
    "USER_DIRECTION_SET_ARTIFACT_TYPE", "DirectionCategory", "DirectionConflictError",
    "DirectionOrigin", "DirectionStage", "DirectionStatus", "DirectionStrength", "GuidanceError",
    "RevisionRequest", "TargetRef", "TargetScope", "UserDirection", "UserDirectionSet",
    "EMPTY_EFFECTIVE_USER_DIRECTION_SET_HASH", "EffectiveUserDirectionSet", "TargetCatalog",
    "load_revision_request", "load_user_direction_set", "persist_revision_request",
    "persist_user_direction", "persist_user_direction_set", "project_effective_directions",
    "resolve_direction_set", "validate_direction_targets", "validate_revision_request",
]
