"""Python-owned validation and deterministic effective direction projection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from short_drama.artifacts import ArtifactError, ArtifactRef, FileArtifactStore, content_hash
from short_drama.foundation._common import artifact_ref_sort_key, require_storage_id

from .models import (
    GUIDANCE_SCHEMA_VERSION,
    USER_DIRECTION_ARTIFACT_TYPE,
    DirectionConflictError,
    DirectionStage,
    DirectionStatus,
    GuidanceError,
    TargetRef,
    TargetScope,
    UserDirection,
    UserDirectionSet,
)


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise GuidanceError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


@dataclass(frozen=True, slots=True)
class TargetCatalog:
    """The already-existing stable-ID universe available to a guidance operation.

    G1 never creates IDs.  A caller supplies the project and targets that the
    current authoritative upstream revision has already materialized.
    """

    project_id: str
    targets: tuple[TargetRef, ...] = ()
    authority_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        require_storage_id(self.project_id, "project_id", GuidanceError)
        targets = tuple(self.targets)
        for target in targets:
            if not isinstance(target, TargetRef) or target.scope is TargetScope.PROJECT:
                raise GuidanceError("TargetCatalog targets must be non-PROJECT TargetRef values")
        targets = tuple(sorted(targets, key=TargetRef.sort_key))
        if len(set(targets)) != len(targets):
            raise GuidanceError("TargetCatalog targets must not contain duplicates")
        object.__setattr__(self, "targets", targets)
        if self.authority_ref is not None and not isinstance(self.authority_ref, ArtifactRef):
            raise GuidanceError("TargetCatalog authority_ref must be an exact ArtifactRef")

    def contains(self, target: TargetRef) -> bool:
        return target.scope is TargetScope.PROJECT or target in self.targets

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "targets": [target.to_dict() for target in self.targets],
            "authority_ref": None if self.authority_ref is None else self.authority_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TargetCatalog":
        value = _exact(value, {"project_id", "targets", "authority_ref"}, cls.__name__)
        if not isinstance(value["targets"], list):
            raise GuidanceError("TargetCatalog targets must be a list")
        try:
            authority_ref = None if value["authority_ref"] is None else ArtifactRef.from_dict(value["authority_ref"])
        except (TypeError, ValueError) as exc:
            raise GuidanceError(f"invalid TargetCatalog authority_ref: {exc}") from exc
        return cls(value["project_id"], tuple(TargetRef.from_dict(item) for item in value["targets"]), authority_ref)


def _target_sort_key(target: TargetRef) -> tuple[object, ...]:
    return target.sort_key()


def _normalize_query_targets(target_refs: Iterable[TargetRef] | None) -> tuple[TargetRef, ...] | None:
    if target_refs is None:
        return None
    refs = tuple(target_refs)
    for ref in refs:
        if not isinstance(ref, TargetRef):
            raise GuidanceError("query target_refs must contain TargetRef values")
    refs = tuple(sorted(refs, key=_target_sort_key))
    if len(set(refs)) != len(refs):
        raise GuidanceError("query target_refs must not contain duplicates")
    return refs


def validate_target_ref(store: FileArtifactStore, catalog: TargetCatalog, target: TargetRef) -> None:
    if not isinstance(store, FileArtifactStore):
        raise GuidanceError("store must be a FileArtifactStore")
    if not isinstance(catalog, TargetCatalog):
        raise GuidanceError("catalog must be a TargetCatalog")
    if not isinstance(target, TargetRef):
        raise GuidanceError("target must be a TargetRef")
    if not catalog.contains(target):
        raise GuidanceError(f"target does not exist in the authoritative target catalog: {target.scope.value}/{target.target_id}")
    if target.scope is not TargetScope.PROJECT:
        if catalog.authority_ref is None:
            raise GuidanceError("non-PROJECT targets require an exact authoritative target catalog ref")
        try:
            store.get_ref(catalog.authority_ref)
        except ArtifactError as exc:
            raise GuidanceError("target catalog authority_ref cannot be resolved exactly") from exc
        # Only A-layer character IDs exist as stable, unmaterialized IDs in G1.
        # Future Scene/Shot/etc. targets must pin their already-created artifact.
        if target.scope is TargetScope.CHARACTER:
            if catalog.authority_ref.artifact_type != "canonical_character_registry":
                raise GuidanceError("CHARACTER targets require a canonical_character_registry authority_ref")
            try:
                from short_drama.story.reconciliation import CanonicalCharacterRegistry

                payload = store.get_ref(catalog.authority_ref).payload
                registry = CanonicalCharacterRegistry.from_dict(payload)  # type: ignore[arg-type]
            except Exception as exc:
                raise GuidanceError("canonical_character_registry authority_ref is not a valid Foundation registry") from exc
            if target.target_id not in {entity.canonical_id for entity in registry.entities}:
                raise GuidanceError("CHARACTER target_id is absent from the exact canonical character registry")
        elif target.artifact_ref is None:
            raise GuidanceError("future/non-character targets require an exact existing ArtifactRef")
    if target.artifact_ref is not None:
        try:
            store.get_ref(target.artifact_ref)
        except ArtifactError as exc:
            raise GuidanceError(f"target ArtifactRef cannot be resolved exactly: {target.artifact_ref!r}") from exc


def validate_direction_targets(store: FileArtifactStore, catalog: TargetCatalog, direction: UserDirection) -> None:
    if not isinstance(direction, UserDirection):
        raise GuidanceError("direction must be a UserDirection")
    for target in direction.target_refs:
        validate_target_ref(store, catalog, target)


def load_user_direction(store: FileArtifactStore, ref: ArtifactRef) -> UserDirection:
    if not isinstance(store, FileArtifactStore):
        raise GuidanceError("store must be a FileArtifactStore")
    if not isinstance(ref, ArtifactRef) or ref.artifact_type != USER_DIRECTION_ARTIFACT_TYPE:
        raise GuidanceError("ref must be an exact user_direction ArtifactRef")
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise GuidanceError(f"user direction ref cannot be resolved exactly: {ref!r}") from exc
    try:
        direction = UserDirection.from_dict(envelope.payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise GuidanceError("persisted user direction payload is invalid") from exc
    if direction.artifact_ref != ref:
        raise GuidanceError("user direction payload identity does not match its exact ArtifactRef")
    return direction


def _verify_revision_lineage(
    store: FileArtifactStore,
    catalog: TargetCatalog,
    direction: UserDirection,
) -> None:
    """Verify every immutable predecessor pin; never infer a current revision."""

    current = direction
    while current.direction_revision > 1:
        predecessors = tuple(
            ref for ref in current.supersedes_direction_refs
            if ref.artifact_id == current.direction_id and ref.revision == current.direction_revision - 1
        )
        if len(predecessors) != 1:
            raise GuidanceError("direction revision must have exactly one exact predecessor supersession ref")
        predecessor = load_user_direction(store, predecessors[0])
        if predecessor.project_id != current.project_id:
            raise GuidanceError("direction revision lineage must remain within one project")
        validate_direction_targets(store, catalog, predecessor)
        current = predecessor


def resolve_direction_set(
    store: FileArtifactStore,
    direction_set: UserDirectionSet,
    catalog: TargetCatalog,
) -> tuple[UserDirection, ...]:
    if not isinstance(direction_set, UserDirectionSet):
        raise GuidanceError("direction_set must be a UserDirectionSet")
    if direction_set.project_id != catalog.project_id:
        raise GuidanceError("direction_set project_id does not match target catalog")
    directions = tuple(load_user_direction(store, ref) for ref in direction_set.direction_refs)
    identities = {(direction.direction_id, direction.direction_revision) for direction in directions}
    if len(identities) != len(directions):
        raise GuidanceError("direction set must not contain duplicate direction ID/revision records")
    for direction in directions:
        if direction.project_id != direction_set.project_id:
            raise GuidanceError("direction project_id does not match UserDirectionSet project_id")
        validate_direction_targets(store, catalog, direction)
        _verify_revision_lineage(store, catalog, direction)
        for ref in (*direction.supersedes_direction_refs, *direction.exception_for_direction_refs):
            related = load_user_direction(store, ref)
            validate_direction_targets(store, catalog, related)
            if related.artifact_ref == direction.artifact_ref:
                raise GuidanceError("a direction cannot supersede or except itself")
    return tuple(sorted(directions, key=lambda direction: artifact_ref_sort_key(direction.artifact_ref)))


def _selected_current_revisions(directions: Iterable[UserDirection]) -> tuple[UserDirection, ...]:
    """Select explicit latest records within each immutable direction revision line."""

    selected: dict[str, UserDirection] = {}
    for direction in directions:
        current = selected.get(direction.direction_id)
        if current is None or direction.direction_revision > current.direction_revision:
            selected[direction.direction_id] = direction
    return tuple(selected.values())


def _has_explicit_exception(left: UserDirection, right: UserDirection) -> bool:
    return right.artifact_ref in left.exception_for_direction_refs or left.artifact_ref in right.exception_for_direction_refs


def _targets_overlap(left: UserDirection, right: UserDirection) -> bool:
    if left.target_scope is TargetScope.PROJECT or right.target_scope is TargetScope.PROJECT:
        return True
    return bool(set(left.target_refs) & set(right.target_refs))


def _hard_conflicts(directions: Iterable[UserDirection]) -> None:
    ordered = tuple(sorted(directions, key=lambda direction: artifact_ref_sort_key(direction.artifact_ref)))
    for index, left in enumerate(ordered):
        if not left.strength.is_hard or left.constraint_key is None:
            continue
        for right in ordered[index + 1 :]:
            if (
                not right.strength.is_hard
                or right.constraint_key != left.constraint_key
                or left.category is not right.category
                or not _targets_overlap(left, right)
                or _has_explicit_exception(left, right)
            ):
                continue
            # The key explicitly identifies a single structured constraint; two
            # different hard declarations for it are ambiguous, including broad
            # versus narrow MUST records.  Never pick one by insertion order.
            if left.strength != right.strength or left.instruction_zh != right.instruction_zh:
                raise DirectionConflictError(
                    "DIRECTION_CONFLICT: unresolved hard directions for "
                    f"{left.constraint_key!r}: {left.artifact_ref!r} vs {right.artifact_ref!r}"
                )


def _matches_query(direction: UserDirection, target_refs: tuple[TargetRef, ...] | None) -> bool:
    if target_refs is None:
        return True
    return direction.target_scope is TargetScope.PROJECT or bool(set(direction.target_refs) & set(target_refs))


def _effective_hash(refs: tuple[ArtifactRef, ...]) -> str:
    return content_hash({"schema_version": GUIDANCE_SCHEMA_VERSION, "direction_refs": [ref.to_dict() for ref in refs]})


EMPTY_EFFECTIVE_USER_DIRECTION_SET_HASH = _effective_hash(())


@dataclass(frozen=True, slots=True)
class EffectiveUserDirectionSet:
    stage: DirectionStage
    target_refs: tuple[TargetRef, ...] = ()
    direction_refs: tuple[ArtifactRef, ...] = ()
    effective_user_direction_set_hash: str = EMPTY_EFFECTIVE_USER_DIRECTION_SET_HASH

    def __post_init__(self) -> None:
        try:
            stage = self.stage if isinstance(self.stage, DirectionStage) else DirectionStage(self.stage)
        except (TypeError, ValueError) as exc:
            raise GuidanceError("stage must be a DirectionStage") from exc
        object.__setattr__(self, "stage", stage)
        targets = _normalize_query_targets(self.target_refs)
        assert targets is not None
        object.__setattr__(self, "target_refs", targets)
        refs = tuple(self.direction_refs)
        for ref in refs:
            if not isinstance(ref, ArtifactRef) or ref.artifact_type != USER_DIRECTION_ARTIFACT_TYPE:
                raise GuidanceError("direction_refs must contain exact user_direction ArtifactRefs")
        refs = tuple(sorted(refs, key=artifact_ref_sort_key))
        if len(set(refs)) != len(refs):
            raise GuidanceError("direction_refs must not contain duplicates")
        object.__setattr__(self, "direction_refs", refs)
        expected = _effective_hash(refs)
        if self.effective_user_direction_set_hash != expected:
            raise GuidanceError("effective_user_direction_set_hash does not match exact member refs")

    @property
    def is_empty(self) -> bool:
        return not self.direction_refs

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "target_refs": [target.to_dict() for target in self.target_refs],
            "direction_refs": [ref.to_dict() for ref in self.direction_refs],
            "effective_user_direction_set_hash": self.effective_user_direction_set_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EffectiveUserDirectionSet":
        value = _exact(value, {"stage", "target_refs", "direction_refs", "effective_user_direction_set_hash"}, cls.__name__)
        if not isinstance(value["target_refs"], list) or not isinstance(value["direction_refs"], list):
            raise GuidanceError("EffectiveUserDirectionSet refs must be lists")
        try:
            return cls(value["stage"], tuple(TargetRef.from_dict(item) for item in value["target_refs"]), tuple(ArtifactRef.from_dict(item) for item in value["direction_refs"]), value["effective_user_direction_set_hash"])
        except (TypeError, ValueError) as exc:
            raise GuidanceError(f"invalid EffectiveUserDirectionSet: {exc}") from exc


def project_effective_directions(
    store: FileArtifactStore,
    direction_set: UserDirectionSet,
    catalog: TargetCatalog,
    *,
    stage: DirectionStage,
    target_refs: Iterable[TargetRef] | None = None,
) -> EffectiveUserDirectionSet:
    """Return the deterministic, stage-specific current guidance projection.

    All set members and their lifecycle provenance are verified before filtering,
    so a dangling or tampered direction cannot be hidden by choosing another stage.
    """

    try:
        stage = stage if isinstance(stage, DirectionStage) else DirectionStage(stage)
    except (TypeError, ValueError) as exc:
        raise GuidanceError("stage must be a DirectionStage") from exc
    query_targets = _normalize_query_targets(target_refs)
    if query_targets is not None:
        for target in query_targets:
            validate_target_ref(store, catalog, target)
    directions = resolve_direction_set(store, direction_set, catalog)
    selected = _selected_current_revisions(directions)
    superseded = {ref for direction in selected for ref in direction.supersedes_direction_refs}
    active = tuple(
        direction for direction in selected
        if direction.status is DirectionStatus.ACTIVE and direction.artifact_ref not in superseded
    )
    relevant = tuple(
        direction for direction in active
        if direction.stage_scope is stage and _matches_query(direction, query_targets)
    )
    _hard_conflicts(relevant)
    refs = tuple(sorted((direction.artifact_ref for direction in relevant), key=artifact_ref_sort_key))
    return EffectiveUserDirectionSet(
        stage=stage,
        target_refs=() if query_targets is None else query_targets,
        direction_refs=refs,
        effective_user_direction_set_hash=_effective_hash(refs),
    )
