"""Persistence adapters for G1 contracts using Foundation immutable artifacts."""
from __future__ import annotations

from short_drama.artifacts import ArtifactError, ArtifactRef, FileArtifactStore, ImmutableArtifactEnvelope

from .models import (
    GUIDANCE_SCHEMA_VERSION,
    REVISION_REQUEST_ARTIFACT_TYPE,
    USER_DIRECTION_ARTIFACT_TYPE,
    USER_DIRECTION_SET_ARTIFACT_TYPE,
    DirectionStage,
    GuidanceError,
    RevisionRequest,
    UserDirection,
    UserDirectionSet,
)
from .projection import TargetCatalog, load_user_direction, resolve_direction_set, validate_target_ref


_ARTIFACT_STAGE_OWNER = {
    "adaptation_target": "B1_ADAPTATION",
    "resolved_adaptation_target": "B1_ADAPTATION",
    "adaptation_plan": "B1_ADAPTATION",
    "episode_screenplay": "B2_SCREENPLAY",
    "screenplay": "B2_SCREENPLAY",
    "production_plan": "B3_PRODUCTION",
    "shot_plan": "B4_SHOT",
    "asset_plan": "B6_ASSET",
}


def _envelope(artifact_type: str, artifact_id: str, revision: int, payload: dict[str, object]) -> ImmutableArtifactEnvelope:
    return ImmutableArtifactEnvelope.create(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=revision,
        schema_version=GUIDANCE_SCHEMA_VERSION,
        payload=payload,  # type: ignore[arg-type]
    )


def persist_user_direction(store: FileArtifactStore, direction: UserDirection) -> ArtifactRef:
    if not isinstance(store, FileArtifactStore) or not isinstance(direction, UserDirection):
        raise GuidanceError("persist_user_direction requires FileArtifactStore and UserDirection")
    ref = store.put(_envelope(USER_DIRECTION_ARTIFACT_TYPE, direction.direction_id, direction.direction_revision, direction.to_dict()))
    if ref != direction.artifact_ref:
        raise GuidanceError("persisted UserDirection does not match its deterministic exact ref")
    return ref


def persist_user_direction_set(
    store: FileArtifactStore,
    direction_set: UserDirectionSet,
    catalog: TargetCatalog,
) -> ArtifactRef:
    if not isinstance(store, FileArtifactStore) or not isinstance(direction_set, UserDirectionSet):
        raise GuidanceError("persist_user_direction_set requires FileArtifactStore and UserDirectionSet")
    # This rejects missing/tampered members before publishing a set that claims
    # them as exact immutable history.
    resolve_direction_set(store, direction_set, catalog)
    ref = store.put(_envelope(USER_DIRECTION_SET_ARTIFACT_TYPE, direction_set.set_id, direction_set.revision, direction_set.to_dict()))
    if ref != direction_set.artifact_ref:
        raise GuidanceError("persisted UserDirectionSet does not match its deterministic exact ref")
    return ref


def load_user_direction_set(store: FileArtifactStore, ref: ArtifactRef) -> UserDirectionSet:
    if not isinstance(store, FileArtifactStore) or not isinstance(ref, ArtifactRef) or ref.artifact_type != USER_DIRECTION_SET_ARTIFACT_TYPE:
        raise GuidanceError("ref must be an exact user_direction_set ArtifactRef")
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise GuidanceError(f"user direction set ref cannot be resolved exactly: {ref!r}") from exc
    try:
        direction_set = UserDirectionSet.from_dict(envelope.payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise GuidanceError("persisted user direction set payload is invalid") from exc
    if direction_set.artifact_ref != ref:
        raise GuidanceError("user direction set payload identity does not match its exact ArtifactRef")
    for direction_ref in direction_set.direction_refs:
        load_user_direction(store, direction_ref)
    return direction_set


def validate_revision_request(store: FileArtifactStore, request: RevisionRequest, catalog: TargetCatalog) -> None:
    if not isinstance(store, FileArtifactStore) or not isinstance(request, RevisionRequest):
        raise GuidanceError("validate_revision_request requires FileArtifactStore and RevisionRequest")
    try:
        store.get_ref(request.target_artifact_ref)
    except ArtifactError as exc:
        raise GuidanceError("RevisionRequest target_artifact_ref cannot be resolved exactly") from exc
    owner_value = _ARTIFACT_STAGE_OWNER.get(request.target_artifact_ref.artifact_type)
    if owner_value is None:
        if request.routed_owner_stage is None:
            raise GuidanceError("unknown revision target artifact type requires explicit routed_owner_stage provenance")
        owner_stage = request.routed_owner_stage
    else:
        owner_stage = DirectionStage(owner_value)
        if request.routed_owner_stage is not None and request.routed_owner_stage is not owner_stage:
            raise GuidanceError("routed_owner_stage disagrees with known target artifact owner")
    if request.requested_stage is not owner_stage:
        raise GuidanceError("requested_stage does not own target_artifact_ref; route to the earliest authoritative owner")
    for target in request.target_refs:
        validate_target_ref(store, catalog, target)
    for ref in request.resulting_direction_refs:
        direction = load_user_direction(store, ref)
        if direction.stage_scope is not request.requested_stage:
            raise GuidanceError("resulting_direction_refs must belong to RevisionRequest requested_stage")


def persist_revision_request(
    store: FileArtifactStore,
    request: RevisionRequest,
    catalog: TargetCatalog,
) -> ArtifactRef:
    validate_revision_request(store, request, catalog)
    ref = store.put(_envelope(REVISION_REQUEST_ARTIFACT_TYPE, request.revision_request_id, 1, request.to_dict()))
    if ref != request.artifact_ref:
        raise GuidanceError("persisted RevisionRequest does not match its deterministic exact ref")
    return ref


def load_revision_request(store: FileArtifactStore, ref: ArtifactRef) -> RevisionRequest:
    if not isinstance(store, FileArtifactStore) or not isinstance(ref, ArtifactRef) or ref.artifact_type != REVISION_REQUEST_ARTIFACT_TYPE:
        raise GuidanceError("ref must be an exact revision_request ArtifactRef")
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise GuidanceError(f"revision request ref cannot be resolved exactly: {ref!r}") from exc
    try:
        request = RevisionRequest.from_dict(envelope.payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise GuidanceError("persisted RevisionRequest payload is invalid") from exc
    if request.artifact_ref != ref:
        raise GuidanceError("revision request payload identity does not match its exact ArtifactRef")
    try:
        store.get_ref(request.target_artifact_ref)
    except ArtifactError as exc:
        raise GuidanceError("RevisionRequest target_artifact_ref cannot be resolved exactly") from exc
    for direction_ref in request.resulting_direction_refs:
        load_user_direction(store, direction_ref)
    return request
