from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from short_drama.artifacts import ArtifactRef, FileArtifactStore, ImmutableArtifactEnvelope
from short_drama.guidance import (
    EMPTY_EFFECTIVE_USER_DIRECTION_SET_HASH,
    DirectionCategory,
    DirectionConflictError,
    DirectionOrigin,
    DirectionStage,
    DirectionStatus,
    DirectionStrength,
    EffectiveUserDirectionSet,
    GuidanceError,
    RevisionRequest,
    TargetCatalog,
    TargetRef,
    TargetScope,
    UserDirection,
    UserDirectionSet,
    load_revision_request,
    load_user_direction_set,
    persist_revision_request,
    persist_user_direction,
    persist_user_direction_set,
    project_effective_directions,
)


HASH = "a" * 64


def _direction(
    direction_id: str,
    *,
    revision: int = 1,
    category: DirectionCategory = DirectionCategory.ADAPTATION_STORY,
    strength: DirectionStrength = DirectionStrength.MUST,
    scope: TargetScope = TargetScope.PROJECT,
    targets: tuple[TargetRef, ...] = (),
    instruction: str = "保留黑色幽默",
    status: DirectionStatus = DirectionStatus.ACTIVE,
    supersedes: tuple[ArtifactRef, ...] = (),
    exceptions: tuple[ArtifactRef, ...] = (),
    key: str | None = "tone",
) -> UserDirection:
    stage = {
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
    }[category]
    return UserDirection(
        direction_id=direction_id,
        direction_revision=revision,
        origin=DirectionOrigin.USER,
        category=category,
        strength=strength,
        stage_scope=stage,
        target_scope=scope,
        target_refs=targets,
        instruction_zh=instruction,
        status=status,
        supersedes_direction_refs=supersedes,
        exception_for_direction_refs=exceptions,
        provenance="reviewer:user-01",
        constraint_key=key,
    )


def _catalog(*targets: TargetRef) -> TargetCatalog:
    return TargetCatalog("project-001", targets)


def _persist_set(
    store: FileArtifactStore, catalog: TargetCatalog, *directions: UserDirection, set_id: str = "guidance-set",
) -> UserDirectionSet:
    refs = tuple(persist_user_direction(store, direction) for direction in directions)
    direction_set = UserDirectionSet("project-001", set_id, 1, refs)
    persist_user_direction_set(store, direction_set, catalog)
    return direction_set


def _artifact(store: FileArtifactStore, artifact_type: str = "screenplay") -> ArtifactRef:
    return store.put(ImmutableArtifactEnvelope.create(
        artifact_type=artifact_type,
        artifact_id="review-target",
        revision=1,
        schema_version=1,
        payload={"value": "exact"},
    ))


def test_empty_guidance_has_a_stable_identity(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    catalog = _catalog()
    direction_set = _persist_set(store, catalog)
    first = project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B1_ADAPTATION)
    second = project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B2_SCREENPLAY)
    assert first.is_empty and second.is_empty
    assert first.effective_user_direction_set_hash == EMPTY_EFFECTIVE_USER_DIRECTION_SET_HASH
    assert second.effective_user_direction_set_hash == EMPTY_EFFECTIVE_USER_DIRECTION_SET_HASH


def test_canonical_ordering_and_hash_are_insertion_independent(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    catalog = _catalog()
    first = _direction("dir-b", instruction="保留柴郡猫", key="cheshire")
    second = _direction("dir-a", instruction="不要爱情线", strength=DirectionStrength.MUST_NOT, key="romance")
    ref_first, ref_second = persist_user_direction(store, first), persist_user_direction(store, second)
    a = UserDirectionSet("project-001", "set-a", 1, (ref_first, ref_second))
    b = UserDirectionSet("project-001", "set-b", 1, (ref_second, ref_first))
    assert a.direction_refs == b.direction_refs
    assert project_effective_directions(store, a, catalog, stage=DirectionStage.B1_ADAPTATION).effective_user_direction_set_hash == project_effective_directions(store, b, catalog, stage=DirectionStage.B1_ADAPTATION).effective_user_direction_set_hash


def test_b4_direction_does_not_change_b1_or_b2_identity(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    catalog = _catalog()
    b1 = _direction("dir-b1")
    b2 = _direction("dir-b2", category=DirectionCategory.DIALOGUE, key="alice-dialogue")
    base = _persist_set(store, catalog, b1, b2)
    before = tuple(project_effective_directions(store, base, catalog, stage=stage).effective_user_direction_set_hash for stage in (DirectionStage.B1_ADAPTATION, DirectionStage.B2_SCREENPLAY))
    b4 = _direction("dir-b4", category=DirectionCategory.CAMERA, instruction="多用特写", key="camera")
    refs = (*base.direction_refs, persist_user_direction(store, b4))
    changed = UserDirectionSet("project-001", "guidance-set", 2, refs)
    persist_user_direction_set(store, changed, catalog)
    after = tuple(project_effective_directions(store, changed, catalog, stage=stage).effective_user_direction_set_hash for stage in (DirectionStage.B1_ADAPTATION, DirectionStage.B2_SCREENPLAY))
    assert before == after
    assert project_effective_directions(store, changed, catalog, stage=DirectionStage.B4_SHOT).is_empty is False


def test_relevant_b1_guidance_changes_only_b1_identity(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    catalog = _catalog()
    base = _persist_set(store, catalog)
    before = project_effective_directions(store, base, catalog, stage=DirectionStage.B1_ADAPTATION)
    b1 = _direction("dir-b1")
    changed = UserDirectionSet("project-001", "guidance-set", 2, (persist_user_direction(store, b1),))
    persist_user_direction_set(store, changed, catalog)
    after = project_effective_directions(store, changed, catalog, stage=DirectionStage.B1_ADAPTATION)
    assert before.effective_user_direction_set_hash != after.effective_user_direction_set_hash


def test_unresolved_hard_conflicts_fail_closed(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    catalog = _catalog()
    required = _direction("must", instruction="保留爱情线", key="romance")
    forbidden = _direction("must-not", instruction="不要爱情线", strength=DirectionStrength.MUST_NOT, key="romance")
    direction_set = _persist_set(store, catalog, required, forbidden)
    with pytest.raises(DirectionConflictError, match="DIRECTION_CONFLICT"):
        project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B1_ADAPTATION)


def test_broad_and_narrow_hard_directions_need_explicit_exception(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    character = TargetRef(TargetScope.CHARACTER, "char-0001")
    catalog = _catalog(character)
    broad = _direction("broad", instruction="Alice 穿蓝白裙", key="alice-wardrobe", category=DirectionCategory.WARDROBE)
    narrow = _direction("narrow", instruction="Alice 穿红裙", key="alice-wardrobe", category=DirectionCategory.WARDROBE, scope=TargetScope.CHARACTER, targets=(character,))
    direction_set = _persist_set(store, catalog, broad, narrow)
    with pytest.raises(DirectionConflictError):
        project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B3_PRODUCTION)
    exception = _direction("narrow-exception", instruction="Alice 穿红裙", key="alice-wardrobe", category=DirectionCategory.WARDROBE, scope=TargetScope.CHARACTER, targets=(character,), exceptions=(broad.artifact_ref,))
    resolved = _persist_set(store, catalog, broad, exception, set_id="guidance-set-exception")
    assert len(project_effective_directions(store, resolved, catalog, stage=DirectionStage.B3_PRODUCTION).direction_refs) == 2


def test_explicit_supersession_withdrawal_and_historical_revisions(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    catalog = _catalog()
    old = _direction("dir-story", instruction="保留茶会", key="tea-party")
    new = _direction("dir-story", revision=2, instruction="压缩茶会", key="tea-party", supersedes=(old.artifact_ref,))
    direction_set = _persist_set(store, catalog, old, new)
    effective = project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B1_ADAPTATION)
    assert effective.direction_refs == (new.artifact_ref,)
    with pytest.raises(FrozenInstanceError):
        old.instruction_zh = "mutate"  # type: ignore[misc]
    assert store.get_ref(old.artifact_ref).ref == old.artifact_ref
    withdrawn = _direction("dir-withdrawn", status=DirectionStatus.WITHDRAWN, key="unused")
    withdrawn_set = _persist_set(store, catalog, withdrawn, set_id="guidance-set-withdrawn")
    assert project_effective_directions(store, withdrawn_set, catalog, stage=DirectionStage.B1_ADAPTATION).is_empty


@pytest.mark.parametrize("bad", [
    ArtifactRef("user_direction", "missing", 1, HASH),
    ArtifactRef("user_direction", "wrong-revision", 2, HASH),
])
def test_missing_and_wrong_revision_direction_refs_are_rejected(tmp_path, bad: ArtifactRef) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    with pytest.raises(GuidanceError):
        persist_user_direction_set(store, UserDirectionSet("project-001", "set", 1, (bad,)), _catalog())


def test_invalid_future_target_and_stage_category_mismatch_are_rejected(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    future_shot = TargetRef(TargetScope.PLANNED_SHOT, "shot-0001")
    direction = _direction("camera", category=DirectionCategory.CAMERA, scope=TargetScope.PLANNED_SHOT, targets=(future_shot,), key="camera")
    with pytest.raises(GuidanceError, match="does not exist"):
        _persist_set(store, _catalog(), direction)
    with pytest.raises(GuidanceError, match="earliest authoritative owner"):
        UserDirection("bad-stage", 1, DirectionOrigin.USER, DirectionCategory.CAMERA, DirectionStrength.MUST, DirectionStage.B1_ADAPTATION, TargetScope.PROJECT, (), "close-up")


def test_revision_request_pins_exact_resolvable_artifact_ref(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    catalog = _catalog()
    target = _artifact(store)
    request = RevisionRequest("rr-001", target, DirectionStage.B2_SCREENPLAY, "缩短台词", (), (), "reviewer:user-01")
    ref = persist_revision_request(store, request, catalog)
    assert load_revision_request(store, ref) == request
    tampered = ArtifactRef(target.artifact_type, target.artifact_id, target.revision, HASH)
    with pytest.raises(GuidanceError, match="cannot be resolved exactly"):
        persist_revision_request(store, RevisionRequest("rr-002", tampered, DirectionStage.B2_SCREENPLAY, "缩短台词", (), (), "reviewer:user-01"), catalog)


def test_serialization_round_trips_and_exact_set_persistence(tmp_path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    character = TargetRef(TargetScope.CHARACTER, "char-0001")
    catalog = _catalog(character)
    direction = _direction("visual", category=DirectionCategory.WARDROBE, scope=TargetScope.CHARACTER, targets=(character,), key="wardrobe")
    assert UserDirection.from_dict(direction.to_dict()) == direction
    direction_set = _persist_set(store, catalog, direction)
    assert UserDirectionSet.from_dict(direction_set.to_dict()) == direction_set
    assert load_user_direction_set(store, direction_set.artifact_ref) == direction_set
    effective = project_effective_directions(store, direction_set, catalog, stage=DirectionStage.B3_PRODUCTION, target_refs=(character,))
    assert EffectiveUserDirectionSet.from_dict(effective.to_dict()) == effective
