"""Immutable, provider-free user-guidance contracts for v1.2 G1.

The objects in this module deliberately describe guidance only.  They do not
resolve an adaptation target, create future scene/shot IDs, or decide whether a
stage complied with a direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from short_drama.artifacts import ArtifactRef, artifact_content_hash
from short_drama.foundation._common import artifact_ref_sort_key, require_storage_id, require_text


USER_DIRECTION_ARTIFACT_TYPE = "user_direction"
USER_DIRECTION_SET_ARTIFACT_TYPE = "user_direction_set"
REVISION_REQUEST_ARTIFACT_TYPE = "revision_request"
GUIDANCE_SCHEMA_VERSION = 1


class GuidanceError(ValueError):
    """A strict G1 contract or projection invariant failed."""


class DirectionConflictError(GuidanceError):
    """Active hard directions conflict without explicit provenance."""


class DirectionOrigin(str, Enum):
    USER = "USER"


class DirectionStrength(str, Enum):
    MUST = "MUST"
    MUST_NOT = "MUST_NOT"
    PREFER = "PREFER"
    AVOID = "AVOID"

    @property
    def is_hard(self) -> bool:
        return self in {self.MUST, self.MUST_NOT}


class DirectionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    WITHDRAWN = "WITHDRAWN"


class DirectionStage(str, Enum):
    B1_ADAPTATION = "B1_ADAPTATION"
    B2_SCREENPLAY = "B2_SCREENPLAY"
    B3_PRODUCTION = "B3_PRODUCTION"
    B4_SHOT = "B4_SHOT"
    B5_RELEASE = "B5_RELEASE"
    B6_ASSET = "B6_ASSET"


class DirectionCategory(str, Enum):
    ADAPTATION_STORY = "ADAPTATION_STORY"
    CHARACTER_NARRATIVE = "CHARACTER_NARRATIVE"
    DIALOGUE = "DIALOGUE"
    SCREENPLAY_ACTION = "SCREENPLAY_ACTION"
    CHARACTER_VISUAL_IDENTITY = "CHARACTER_VISUAL_IDENTITY"
    WARDROBE = "WARDROBE"
    APPEARANCE_STATE = "APPEARANCE_STATE"
    PERFORMANCE = "PERFORMANCE"
    STAGING = "STAGING"
    SPATIAL_POSITION = "SPATIAL_POSITION"
    SHOT_DECOMPOSITION = "SHOT_DECOMPOSITION"
    CAMERA = "CAMERA"
    COMPOSITION = "COMPOSITION"
    SHOT_CONTINUITY = "SHOT_CONTINUITY"
    ASSET_SELECTION = "ASSET_SELECTION"
    ASSET_REJECTION = "ASSET_REJECTION"


class TargetScope(str, Enum):
    PROJECT = "PROJECT"
    CHARACTER = "CHARACTER"
    EPISODE = "EPISODE"
    SCENE = "SCENE"
    SCREENPLAY_ELEMENT = "SCREENPLAY_ELEMENT"
    PRODUCTION_UNIT = "PRODUCTION_UNIT"
    PRODUCTION_REQUIREMENT = "PRODUCTION_REQUIREMENT"
    PLANNED_SHOT = "PLANNED_SHOT"
    UTTERANCE = "UTTERANCE"
    ASSET_REQUIREMENT = "ASSET_REQUIREMENT"
    ASSET_BINDING = "ASSET_BINDING"


_CATEGORY_OWNER = {
    DirectionCategory.ADAPTATION_STORY: DirectionStage.B1_ADAPTATION,
    DirectionCategory.CHARACTER_NARRATIVE: DirectionStage.B1_ADAPTATION,
    DirectionCategory.DIALOGUE: DirectionStage.B2_SCREENPLAY,
    DirectionCategory.SCREENPLAY_ACTION: DirectionStage.B2_SCREENPLAY,
    DirectionCategory.CHARACTER_VISUAL_IDENTITY: DirectionStage.B3_PRODUCTION,
    DirectionCategory.WARDROBE: DirectionStage.B3_PRODUCTION,
    DirectionCategory.APPEARANCE_STATE: DirectionStage.B3_PRODUCTION,
    DirectionCategory.PERFORMANCE: DirectionStage.B3_PRODUCTION,
    DirectionCategory.STAGING: DirectionStage.B3_PRODUCTION,
    DirectionCategory.SPATIAL_POSITION: DirectionStage.B3_PRODUCTION,
    DirectionCategory.SHOT_DECOMPOSITION: DirectionStage.B4_SHOT,
    DirectionCategory.CAMERA: DirectionStage.B4_SHOT,
    DirectionCategory.COMPOSITION: DirectionStage.B4_SHOT,
    DirectionCategory.SHOT_CONTINUITY: DirectionStage.B4_SHOT,
    DirectionCategory.ASSET_SELECTION: DirectionStage.B6_ASSET,
    DirectionCategory.ASSET_REJECTION: DirectionStage.B6_ASSET,
}

_STAGE_TARGET_SCOPES = {
    DirectionStage.B1_ADAPTATION: {TargetScope.PROJECT, TargetScope.CHARACTER, TargetScope.EPISODE},
    DirectionStage.B2_SCREENPLAY: {TargetScope.PROJECT, TargetScope.CHARACTER, TargetScope.EPISODE, TargetScope.SCENE, TargetScope.SCREENPLAY_ELEMENT, TargetScope.UTTERANCE},
    DirectionStage.B3_PRODUCTION: {TargetScope.PROJECT, TargetScope.CHARACTER, TargetScope.EPISODE, TargetScope.SCENE, TargetScope.PRODUCTION_UNIT, TargetScope.PRODUCTION_REQUIREMENT},
    DirectionStage.B4_SHOT: {TargetScope.PROJECT, TargetScope.EPISODE, TargetScope.SCENE, TargetScope.PLANNED_SHOT},
    DirectionStage.B5_RELEASE: {TargetScope.PROJECT, TargetScope.EPISODE, TargetScope.SCENE},
    DirectionStage.B6_ASSET: {TargetScope.PROJECT, TargetScope.CHARACTER, TargetScope.ASSET_REQUIREMENT, TargetScope.ASSET_BINDING},
}


def owner_stage_for_category(category: DirectionCategory) -> DirectionStage:
    """Return the frozen earliest owner for one structured guidance category."""

    return _CATEGORY_OWNER[_enum(category, DirectionCategory, "category")]


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        raise GuidanceError(f"{name} must be one of: {', '.join(item.value for item in enum)}") from exc


def _revision(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GuidanceError(f"{name} must be an integer >= 1")
    return value


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise GuidanceError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _ref(value: Any, name: str, *, artifact_type: str | None = None) -> ArtifactRef:
    if not isinstance(value, ArtifactRef):
        raise GuidanceError(f"{name} must be an ArtifactRef")
    if artifact_type is not None and value.artifact_type != artifact_type:
        raise GuidanceError(f"{name} must be an exact {artifact_type} ArtifactRef")
    return value


def _direction_refs(value: Iterable[ArtifactRef], name: str) -> tuple[ArtifactRef, ...]:
    refs = tuple(value)
    for ref in refs:
        _ref(ref, name, artifact_type=USER_DIRECTION_ARTIFACT_TYPE)
    ordered = tuple(sorted(refs, key=artifact_ref_sort_key))
    if len(set(ordered)) != len(ordered):
        raise GuidanceError(f"{name} must not contain duplicate refs")
    return ordered


@dataclass(frozen=True, slots=True)
class TargetRef:
    """An exact existing stable target, optionally pinned to its artifact revision."""

    scope: TargetScope
    target_id: str
    artifact_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _enum(self.scope, TargetScope, "scope"))
        require_storage_id(self.target_id, "target_id", GuidanceError)
        if self.artifact_ref is not None:
            _ref(self.artifact_ref, "artifact_ref")

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope.value,
            "target_id": self.target_id,
            "artifact_ref": None if self.artifact_ref is None else self.artifact_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TargetRef":
        value = _exact(value, {"scope", "target_id", "artifact_ref"}, cls.__name__)
        ref_value = value["artifact_ref"]
        try:
            ref = None if ref_value is None else ArtifactRef.from_dict(ref_value)
        except (TypeError, ValueError) as exc:
            raise GuidanceError(f"invalid TargetRef artifact_ref: {exc}") from exc
        return cls(value["scope"], value["target_id"], ref)

    def sort_key(self) -> tuple[object, ...]:
        return (self.scope.value, self.target_id, () if self.artifact_ref is None else artifact_ref_sort_key(self.artifact_ref))


def _targets(value: Iterable[TargetRef], name: str, *, scope: TargetScope) -> tuple[TargetRef, ...]:
    refs = tuple(value)
    if scope is TargetScope.PROJECT:
        if refs:
            raise GuidanceError("PROJECT target_scope must not have target_refs")
        return ()
    if not refs:
        raise GuidanceError("non-PROJECT target_scope requires at least one exact target_ref")
    for ref in refs:
        if not isinstance(ref, TargetRef):
            raise GuidanceError(f"{name} must contain TargetRef values")
        if ref.scope is not scope:
            raise GuidanceError("target_refs must use the declared target_scope")
    ordered = tuple(sorted(refs, key=TargetRef.sort_key))
    if len(set(ordered)) != len(ordered):
        raise GuidanceError(f"{name} must not contain duplicates")
    return ordered


@dataclass(frozen=True, slots=True)
class UserDirection:
    direction_id: str
    project_id: str
    direction_revision: int
    origin: DirectionOrigin
    category: DirectionCategory
    strength: DirectionStrength
    stage_scope: DirectionStage
    target_scope: TargetScope
    target_refs: tuple[TargetRef, ...]
    instruction_zh: str
    status: DirectionStatus = DirectionStatus.ACTIVE
    supersedes_direction_refs: tuple[ArtifactRef, ...] = ()
    exception_for_direction_refs: tuple[ArtifactRef, ...] = ()
    provenance: str = "USER"
    constraint_key: str | None = None

    def __post_init__(self) -> None:
        require_storage_id(self.direction_id, "direction_id", GuidanceError)
        require_storage_id(self.project_id, "project_id", GuidanceError)
        _revision(self.direction_revision, "direction_revision")
        object.__setattr__(self, "origin", _enum(self.origin, DirectionOrigin, "origin"))
        object.__setattr__(self, "category", _enum(self.category, DirectionCategory, "category"))
        object.__setattr__(self, "strength", _enum(self.strength, DirectionStrength, "strength"))
        object.__setattr__(self, "stage_scope", _enum(self.stage_scope, DirectionStage, "stage_scope"))
        object.__setattr__(self, "target_scope", _enum(self.target_scope, TargetScope, "target_scope"))
        object.__setattr__(self, "status", _enum(self.status, DirectionStatus, "status"))
        if self.origin is not DirectionOrigin.USER:
            raise GuidanceError("G1 only accepts USER direction origin")
        if _CATEGORY_OWNER[self.category] is not self.stage_scope:
            raise GuidanceError("stage_scope does not own category; route to the earliest authoritative owner")
        if self.target_scope not in _STAGE_TARGET_SCOPES[self.stage_scope]:
            raise GuidanceError("target_scope is not owned by stage_scope; route to the earliest authoritative owner")
        object.__setattr__(self, "target_refs", _targets(self.target_refs, "target_refs", scope=self.target_scope))
        require_text(self.instruction_zh, "instruction_zh", GuidanceError)
        require_text(self.provenance, "provenance", GuidanceError)
        if self.strength.is_hard and self.constraint_key is None:
            raise GuidanceError("hard directions require a machine-checkable constraint_key")
        if self.constraint_key is not None:
            require_text(self.constraint_key, "constraint_key", GuidanceError)
        object.__setattr__(self, "supersedes_direction_refs", _direction_refs(self.supersedes_direction_refs, "supersedes_direction_refs"))
        object.__setattr__(self, "exception_for_direction_refs", _direction_refs(self.exception_for_direction_refs, "exception_for_direction_refs"))
        if self.direction_revision > 1:
            if not any(
                ref.artifact_id == self.direction_id and ref.revision == self.direction_revision - 1
                for ref in self.supersedes_direction_refs
            ):
                raise GuidanceError("new direction revisions must explicitly supersede their exact predecessor")
        if self.status in {DirectionStatus.WITHDRAWN, DirectionStatus.SUPERSEDED} and self.direction_revision == 1:
            raise GuidanceError("WITHDRAWN/SUPERSEDED lifecycle records must audit the prior direction revision")

    @property
    def artifact_ref(self) -> ArtifactRef:
        return ArtifactRef(USER_DIRECTION_ARTIFACT_TYPE, self.direction_id, self.direction_revision, self.content_hash)

    @property
    def content_hash(self) -> str:
        return artifact_content_hash(schema_version=GUIDANCE_SCHEMA_VERSION, payload=self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "direction_id": self.direction_id,
            "project_id": self.project_id,
            "direction_revision": self.direction_revision,
            "origin": self.origin.value,
            "category": self.category.value,
            "strength": self.strength.value,
            "stage_scope": self.stage_scope.value,
            "target_scope": self.target_scope.value,
            "target_refs": [ref.to_dict() for ref in self.target_refs],
            "instruction_zh": self.instruction_zh,
            "status": self.status.value,
            "supersedes_direction_refs": [ref.to_dict() for ref in self.supersedes_direction_refs],
            "exception_for_direction_refs": [ref.to_dict() for ref in self.exception_for_direction_refs],
            "provenance": self.provenance,
            "constraint_key": self.constraint_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserDirection":
        value = _exact(value, {
            "direction_id", "project_id", "direction_revision", "origin", "category", "strength", "stage_scope",
            "target_scope", "target_refs", "instruction_zh", "status", "supersedes_direction_refs",
            "exception_for_direction_refs", "provenance", "constraint_key",
        }, cls.__name__)
        if not isinstance(value["target_refs"], list) or not isinstance(value["supersedes_direction_refs"], list) or not isinstance(value["exception_for_direction_refs"], list):
            raise GuidanceError("direction refs must be lists")
        try:
            return cls(
                direction_id=value["direction_id"], project_id=value["project_id"], direction_revision=value["direction_revision"], origin=value["origin"],
                category=value["category"], strength=value["strength"], stage_scope=value["stage_scope"],
                target_scope=value["target_scope"], target_refs=tuple(TargetRef.from_dict(item) for item in value["target_refs"]),
                instruction_zh=value["instruction_zh"], status=value["status"],
                supersedes_direction_refs=tuple(ArtifactRef.from_dict(item) for item in value["supersedes_direction_refs"]),
                exception_for_direction_refs=tuple(ArtifactRef.from_dict(item) for item in value["exception_for_direction_refs"]),
                provenance=value["provenance"], constraint_key=value["constraint_key"],
            )
        except (TypeError, ValueError) as exc:
            raise GuidanceError(f"invalid UserDirection: {exc}") from exc


@dataclass(frozen=True, slots=True)
class UserDirectionSet:
    project_id: str
    set_id: str
    revision: int
    direction_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        require_storage_id(self.project_id, "project_id", GuidanceError)
        require_storage_id(self.set_id, "set_id", GuidanceError)
        _revision(self.revision, "revision")
        object.__setattr__(self, "direction_refs", _direction_refs(self.direction_refs, "direction_refs"))

    @property
    def artifact_ref(self) -> ArtifactRef:
        return ArtifactRef(USER_DIRECTION_SET_ARTIFACT_TYPE, self.set_id, self.revision, self.content_hash)

    @property
    def content_hash(self) -> str:
        return artifact_content_hash(schema_version=GUIDANCE_SCHEMA_VERSION, payload=self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {"project_id": self.project_id, "set_id": self.set_id, "revision": self.revision, "direction_refs": [ref.to_dict() for ref in self.direction_refs]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserDirectionSet":
        value = _exact(value, {"project_id", "set_id", "revision", "direction_refs"}, cls.__name__)
        if not isinstance(value["direction_refs"], list):
            raise GuidanceError("direction_refs must be a list")
        try:
            return cls(value["project_id"], value["set_id"], value["revision"], tuple(ArtifactRef.from_dict(item) for item in value["direction_refs"]))
        except (TypeError, ValueError) as exc:
            raise GuidanceError(f"invalid UserDirectionSet: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RevisionRequest:
    revision_request_id: str
    target_artifact_ref: ArtifactRef
    requested_stage: DirectionStage
    instruction_zh: str
    target_refs: tuple[TargetRef, ...]
    resulting_direction_refs: tuple[ArtifactRef, ...]
    reviewer_provenance: str
    routed_owner_stage: DirectionStage | None = None
    requested_category: DirectionCategory | None = None

    def __post_init__(self) -> None:
        require_storage_id(self.revision_request_id, "revision_request_id", GuidanceError)
        _ref(self.target_artifact_ref, "target_artifact_ref")
        object.__setattr__(self, "requested_stage", _enum(self.requested_stage, DirectionStage, "requested_stage"))
        if self.routed_owner_stage is not None:
            object.__setattr__(self, "routed_owner_stage", _enum(self.routed_owner_stage, DirectionStage, "routed_owner_stage"))
        if self.requested_category is not None:
            object.__setattr__(self, "requested_category", _enum(self.requested_category, DirectionCategory, "requested_category"))
        require_text(self.instruction_zh, "instruction_zh", GuidanceError)
        require_text(self.reviewer_provenance, "reviewer_provenance", GuidanceError)
        refs = tuple(self.target_refs)
        for ref in refs:
            if not isinstance(ref, TargetRef):
                raise GuidanceError("target_refs must contain TargetRef values")
        refs = tuple(sorted(refs, key=TargetRef.sort_key))
        if len(set(refs)) != len(refs):
            raise GuidanceError("target_refs must not contain duplicates")
        object.__setattr__(self, "target_refs", refs)
        object.__setattr__(self, "resulting_direction_refs", _direction_refs(self.resulting_direction_refs, "resulting_direction_refs"))

    @property
    def artifact_ref(self) -> ArtifactRef:
        return ArtifactRef(REVISION_REQUEST_ARTIFACT_TYPE, self.revision_request_id, 1, self.content_hash)

    @property
    def content_hash(self) -> str:
        return artifact_content_hash(schema_version=GUIDANCE_SCHEMA_VERSION, payload=self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "revision_request_id": self.revision_request_id,
            "target_artifact_ref": self.target_artifact_ref.to_dict(),
            "requested_stage": self.requested_stage.value,
            "instruction_zh": self.instruction_zh,
            "target_refs": [ref.to_dict() for ref in self.target_refs],
            "resulting_direction_refs": [ref.to_dict() for ref in self.resulting_direction_refs],
            "reviewer_provenance": self.reviewer_provenance,
            "routed_owner_stage": None if self.routed_owner_stage is None else self.routed_owner_stage.value,
            "requested_category": None if self.requested_category is None else self.requested_category.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RevisionRequest":
        value = _exact(value, {"revision_request_id", "target_artifact_ref", "requested_stage", "instruction_zh", "target_refs", "resulting_direction_refs", "reviewer_provenance", "routed_owner_stage", "requested_category"}, cls.__name__)
        if not isinstance(value["target_refs"], list) or not isinstance(value["resulting_direction_refs"], list):
            raise GuidanceError("RevisionRequest refs must be lists")
        try:
            return cls(value["revision_request_id"], ArtifactRef.from_dict(value["target_artifact_ref"]), value["requested_stage"], value["instruction_zh"], tuple(TargetRef.from_dict(item) for item in value["target_refs"]), tuple(ArtifactRef.from_dict(item) for item in value["resulting_direction_refs"]), value["reviewer_provenance"], value["routed_owner_stage"], value["requested_category"])
        except (TypeError, ValueError) as exc:
            raise GuidanceError(f"invalid RevisionRequest: {exc}") from exc
