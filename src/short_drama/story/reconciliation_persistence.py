"""v1.2 A4D — reconciliation persistence, reuse, and publication.

This module is the A4D *persistence / reuse* layer. It sits on top of the A4A
domain models, the A4B plan, the A4C semantic result, and the A4D
finalization + validation slices. It owns:

  * deterministic A4 artifact identity (base + six outputs + A4 validation
    report + CURRENT pointer);
  * immutable A4 persistence + fail-closed typed loaders;
  * the exact deterministic A4 ``ValidationReport`` (findings=(), exact
    upstream lineage + all six A4 outputs) and its compare-and-set ordering;
  * current-only A4 reuse: resolve the exact CURRENT EntityMap, *fully* verify
    it before comparing the requested (A3 input, A4 semantic identity), then
    either reuse it or publish a new revision under the same logical identity.

The public service mirrors the A3C two-phase contract so A4E can short-circuit
*before* the provider call:

  * :meth:`ReconciliationPersistenceService.try_reuse_current` — pre-generation;
    requires only the backend-neutral identity material deterministically
    available before the LLM call (``a3_input`` + ``semantic_identity`` +
    ``reconciliation_profile_id``).
  * :meth:`ReconciliationPersistenceService.publish_validated` — post-generation;
    receives the in-memory :class:`ReconciliationFinalizationResult`, validates
    it (zero blocking findings), persists an immutable run revision, publishes
    the exact matching PASS A4 ValidationReport, and compare-and-set moves
    CURRENT.

It is deliberately backend-neutral: changing only the provider / model routing
does NOT invalidate A4 reuse (the concrete backend is provenance-only). Corrupt
CURRENT state always fails closed (it is never silently repaired). Orphan
revisions are skipped. A4D does NOT add a CLI or a real-novel acceptance test
(A4E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from short_drama.artifacts import (
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactRef,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
)
from short_drama.foundation import (
    VALIDATION_REPORT_ARTIFACT_TYPE,
    FilePointerStore,
    LineageRef,
    PointerKind,
    PointerNotFoundError,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    load_validation_report,
    persist_validation_report,
)
from short_drama.io import load_json
from short_drama.paths import SCHEMAS_DIR

from .errors import (
    ReconciliationPlanningError,
    StoryIntegrityError,
    StoryPersistenceError,
)
from .reconciliation import (
    CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
    CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
    ENTITY_MAP_SCHEMA_VERSION,
    RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
    UNRESOLVED_ENTITY_SET_SCHEMA_VERSION,
    A3InputIdentity,
    A4SemanticIdentity,
    CandidateEntityIndex,
    CanonicalCharacterRegistry,
    CanonicalLocationRegistry,
    EntityMap,
    EntityReconciliationProfile,
    ReconciliationDecision,
    ReconciliationDecisionSet,
    UnresolvedEntitySet,
)
from .reconciliation_finalization import (
    ReconciliationFinalizationError,
    ReconciliationFinalizationResult,
    build_identity_graph,
    derive_reconciliation_outputs,
)
from .reconciliation_planning import (
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_MUST_NOT_MERGE,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    plan_candidate_index_v1,
    validate_candidate_index_source_order,
)
from .reconciliation_semantic import compute_llm_decision_id, llm_reason_code
from .reconciliation_validation import validate_finalization


# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------

CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE = "candidate_entity_index"
RECONCILIATION_DECISION_SET_ARTIFACT_TYPE = "reconciliation_decision_set"
CANONICAL_CHARACTER_REGISTRY_ARTIFACT_TYPE = "canonical_character_registry"
CANONICAL_LOCATION_REGISTRY_ARTIFACT_TYPE = "canonical_location_registry"
UNRESOLVED_ENTITY_SET_ARTIFACT_TYPE = "unresolved_entity_set"
ENTITY_MAP_ARTIFACT_TYPE = "entity_map"


# ---------------------------------------------------------------------------
# Artifact identity (frozen A-I4 contract, section 25)
# ---------------------------------------------------------------------------


def a4_base_artifact_id(
    project_id: str, document_id: str, reconciliation_profile_id: str
) -> str:
    """Deterministic A4 base identity.

    ``<project_id>.<document_id>.a4.<reconciliation_profile_id>``
    """
    return f"{project_id}.{document_id}.a4.{reconciliation_profile_id}"


def candidate_entity_index_artifact_id(base: str) -> str:
    return f"{base}.candidate-index"


def reconciliation_decision_set_artifact_id(base: str) -> str:
    return f"{base}.decisions"


def canonical_character_registry_artifact_id(base: str) -> str:
    return f"{base}.characters"


def canonical_location_registry_artifact_id(base: str) -> str:
    return f"{base}.locations"


def unresolved_entity_set_artifact_id(base: str) -> str:
    return f"{base}.unresolved"


def entity_map_artifact_id(base: str) -> str:
    return f"{base}.entity-map"


def a4_validation_artifact_id(base: str) -> str:
    return f"{base}.entity-map.a4-validation"


def a4_pointer_id(project_id: str, document_id: str, reconciliation_profile_id: str) -> str:
    """Deterministic A4 CURRENT-pointer identity.

    ``<project_id>.a4.<document_id>.<reconciliation_profile_id>``

    A4 has no approval gate, so the pointer kind is ``CURRENT`` (never
    ``CURRENT_APPROVED``).
    """
    return f"{project_id}.a4.{document_id}.{reconciliation_profile_id}"


def _a4_revision_slots(base: str) -> tuple[tuple[str, str], ...]:
    """The seven A4 artifact slots that share one run revision."""
    return (
        (CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE, candidate_entity_index_artifact_id(base)),
        (RECONCILIATION_DECISION_SET_ARTIFACT_TYPE, reconciliation_decision_set_artifact_id(base)),
        (CANONICAL_CHARACTER_REGISTRY_ARTIFACT_TYPE, canonical_character_registry_artifact_id(base)),
        (CANONICAL_LOCATION_REGISTRY_ARTIFACT_TYPE, canonical_location_registry_artifact_id(base)),
        (UNRESOLVED_ENTITY_SET_ARTIFACT_TYPE, unresolved_entity_set_artifact_id(base)),
        (ENTITY_MAP_ARTIFACT_TYPE, entity_map_artifact_id(base)),
        (VALIDATION_REPORT_ARTIFACT_TYPE, a4_validation_artifact_id(base)),
    )


# ---------------------------------------------------------------------------
# Revision allocation (shared run revision, skip orphans)
# ---------------------------------------------------------------------------


def next_a4_revision(
    store: FileArtifactStore,
    *,
    base: str,
    current_entity_map_ref: ArtifactRef | None,
) -> int:
    """Allocate the next shared run revision across all seven A4 artifact slots.

    A run revision must be free for every A4 artifact identity (entity map,
    five outputs, A4 validation report). Collisions with already-existing
    historical revisions (including orphans left by a failed publication) are
    skipped; an existing revision is never overwritten.
    """
    start = 1 if current_entity_map_ref is None else current_entity_map_ref.revision + 1
    revision = max(1, start)
    while True:
        occupied = False
        for artifact_type, artifact_id in _a4_revision_slots(base):
            try:
                store.get(artifact_type, artifact_id, revision)
            except ArtifactNotFoundError:
                continue
            except ArtifactError as exc:
                raise StoryPersistenceError(
                    f"cannot inspect {artifact_type}/{artifact_id} "
                    f"revision {revision}"
                ) from exc
            occupied = True
            break
        if not occupied:
            return revision
        revision += 1


# ---------------------------------------------------------------------------
# Immutable persistence
# ---------------------------------------------------------------------------


def _envelope(
    artifact_type: str,
    artifact_id: str,
    revision: int,
    schema_version: int,
    payload: dict[str, Any],
    label: str,
):
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=revision,
        schema_version=schema_version,
        payload=payload,
    )
    try:
        return envelope
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(f"failed to build {label} envelope: {exc}") from exc


def persist_candidate_entity_index(
    store: FileArtifactStore, index: CandidateEntityIndex, *, artifact_id: str, revision: int
) -> ArtifactRef:
    if not isinstance(index, CandidateEntityIndex):
        raise StoryIntegrityError("index must be a CandidateEntityIndex")
    envelope = _envelope(
        CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        index.to_dict(),
        "CandidateEntityIndex",
    )
    return _put(store, envelope, "CandidateEntityIndex")


def persist_reconciliation_decision_set(
    store: FileArtifactStore, decision_set: ReconciliationDecisionSet, *, artifact_id: str, revision: int
) -> ArtifactRef:
    if not isinstance(decision_set, ReconciliationDecisionSet):
        raise StoryIntegrityError("decision_set must be a ReconciliationDecisionSet")
    envelope = _envelope(
        RECONCILIATION_DECISION_SET_ARTIFACT_TYPE,
        artifact_id,
        revision,
        RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
        decision_set.to_dict(),
        "ReconciliationDecisionSet",
    )
    return _put(store, envelope, "ReconciliationDecisionSet")


def persist_canonical_character_registry(
    store: FileArtifactStore, registry: CanonicalCharacterRegistry, *, artifact_id: str, revision: int
) -> ArtifactRef:
    if not isinstance(registry, CanonicalCharacterRegistry):
        raise StoryIntegrityError("registry must be a CanonicalCharacterRegistry")
    envelope = _envelope(
        CANONICAL_CHARACTER_REGISTRY_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
        registry.to_dict(),
        "CanonicalCharacterRegistry",
    )
    return _put(store, envelope, "CanonicalCharacterRegistry")


def persist_canonical_location_registry(
    store: FileArtifactStore, registry: CanonicalLocationRegistry, *, artifact_id: str, revision: int
) -> ArtifactRef:
    if not isinstance(registry, CanonicalLocationRegistry):
        raise StoryIntegrityError("registry must be a CanonicalLocationRegistry")
    envelope = _envelope(
        CANONICAL_LOCATION_REGISTRY_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
        registry.to_dict(),
        "CanonicalLocationRegistry",
    )
    return _put(store, envelope, "CanonicalLocationRegistry")


def persist_unresolved_entity_set(
    store: FileArtifactStore, entity_set: UnresolvedEntitySet, *, artifact_id: str, revision: int
) -> ArtifactRef:
    if not isinstance(entity_set, UnresolvedEntitySet):
        raise StoryIntegrityError("entity_set must be an UnresolvedEntitySet")
    envelope = _envelope(
        UNRESOLVED_ENTITY_SET_ARTIFACT_TYPE,
        artifact_id,
        revision,
        UNRESOLVED_ENTITY_SET_SCHEMA_VERSION,
        entity_set.to_dict(),
        "UnresolvedEntitySet",
    )
    return _put(store, envelope, "UnresolvedEntitySet")


def persist_entity_map(
    store: FileArtifactStore, entity_map: EntityMap, *, artifact_id: str, revision: int
) -> ArtifactRef:
    if not isinstance(entity_map, EntityMap):
        raise StoryIntegrityError("entity_map must be an EntityMap")
    envelope = _envelope(
        ENTITY_MAP_ARTIFACT_TYPE,
        artifact_id,
        revision,
        ENTITY_MAP_SCHEMA_VERSION,
        entity_map.to_dict(),
        "EntityMap",
    )
    return _put(store, envelope, "EntityMap")


def _put(store: FileArtifactStore, envelope: ImmutableArtifactEnvelope, label: str) -> ArtifactRef:
    try:
        return store.put(envelope)
    except ArtifactError as exc:
        raise StoryPersistenceError(f"failed to persist {label}: {exc}") from exc


# ---------------------------------------------------------------------------
# Schema gate (single source of truth: tracked JSON Schema)
# ---------------------------------------------------------------------------


def _validate_schema(payload: dict[str, Any], schema_filename: str, label: str) -> None:
    schema = load_json(SCHEMAS_DIR / schema_filename)
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(f"invalid {label} schema: {exc}") from exc
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda e: (tuple(str(p) for p in e.absolute_path), e.message),
    )
    if errors:
        error = errors[0]
        path = "/".join(str(p) for p in error.absolute_path) or "<root>"
        raise StoryIntegrityError(
            f"{label} canonical form fails {schema_filename} at {path}: {error.message}"
        )


# ---------------------------------------------------------------------------
# Fail-closed typed loaders
# ---------------------------------------------------------------------------


def _load_typed(
    store: FileArtifactStore,
    ref: ArtifactRef,
    *,
    artifact_type: str,
    schema_version: int,
    expected_artifact_id: str,
    schema_filename: str,
    label: str,
    loader,
):
    if not isinstance(ref, ArtifactRef) or ref.artifact_type != artifact_type:
        raise StoryIntegrityError(f"{label} ref has the wrong artifact_type")
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise StoryIntegrityError(f"failed to resolve {label}: {ref!r}") from exc
    if envelope.schema_version != schema_version:
        raise StoryIntegrityError(
            f"unsupported {label} schema_version: {envelope.schema_version}; "
            f"supported={schema_version}"
        )
    payload = envelope.payload
    if not isinstance(payload, dict):
        raise StoryIntegrityError(f"persisted {label} payload must be an object")
    try:
        typed = loader(payload)
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(f"invalid persisted {label}: {exc}") from exc
    if typed.to_dict() != payload:
        raise StoryIntegrityError(f"persisted {label} payload is not in canonical semantic form")
    if ref.artifact_id != expected_artifact_id:
        raise StoryIntegrityError(f"{label} artifact_id does not match its logical identity")
    _validate_schema(payload, schema_filename, label)
    return typed


def load_candidate_entity_index(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> CandidateEntityIndex:
    return _load_typed(
        store,
        ref,
        artifact_type=CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE,
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="candidate-entity-index.schema.json",
        label="CandidateEntityIndex",
        loader=CandidateEntityIndex.from_dict,
    )


def load_reconciliation_decision_set(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> ReconciliationDecisionSet:
    return _load_typed(
        store,
        ref,
        artifact_type=RECONCILIATION_DECISION_SET_ARTIFACT_TYPE,
        schema_version=RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="reconciliation-decision-set.schema.json",
        label="ReconciliationDecisionSet",
        loader=ReconciliationDecisionSet.from_dict,
    )


def load_canonical_character_registry(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> CanonicalCharacterRegistry:
    return _load_typed(
        store,
        ref,
        artifact_type=CANONICAL_CHARACTER_REGISTRY_ARTIFACT_TYPE,
        schema_version=CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="canonical-character-registry.schema.json",
        label="CanonicalCharacterRegistry",
        loader=CanonicalCharacterRegistry.from_dict,
    )


def load_canonical_location_registry(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> CanonicalLocationRegistry:
    return _load_typed(
        store,
        ref,
        artifact_type=CANONICAL_LOCATION_REGISTRY_ARTIFACT_TYPE,
        schema_version=CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="canonical-location-registry.schema.json",
        label="CanonicalLocationRegistry",
        loader=CanonicalLocationRegistry.from_dict,
    )


def load_unresolved_entity_set(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> UnresolvedEntitySet:
    return _load_typed(
        store,
        ref,
        artifact_type=UNRESOLVED_ENTITY_SET_ARTIFACT_TYPE,
        schema_version=UNRESOLVED_ENTITY_SET_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="unresolved-entity-set.schema.json",
        label="UnresolvedEntitySet",
        loader=UnresolvedEntitySet.from_dict,
    )


def load_entity_map(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> EntityMap:
    """Fail-closed EntityMap v2 loader.

    Requires ``schema_version == 2`` and a present, well-formed
    ``semantic_identity`` (via the typed model + the v2 schema gate).
    """
    return _load_typed(
        store,
        ref,
        artifact_type=ENTITY_MAP_ARTIFACT_TYPE,
        schema_version=ENTITY_MAP_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="entity-map.schema.json",
        label="EntityMap",
        loader=EntityMap.from_dict,
    )


# ---------------------------------------------------------------------------
# A4 ValidationReport (frozen contract sections 35/37)
# ---------------------------------------------------------------------------


def build_a4_validation_report(
    entity_map: EntityMap, entity_map_ref: ArtifactRef
) -> ValidationReport:
    """The exact deterministic PASS A4 ValidationReport for an EntityMap.

    Lineage (all roles deterministic): the exact upstream A3 refs (source
    document, chunk manifest, and each candidate extraction by deterministic
    ordinal ``candidate_extraction_NNNN``) plus all six A4 outputs (candidate
    entity index, decision set, both canonical registries, unresolved set, and
    the entity map itself). The successful report has ``findings = ()``.
    """
    a3 = entity_map.a3_input
    refs: list[LineageRef] = [
        LineageRef("source_document", a3.source_document_ref),
        LineageRef("chunk_manifest", a3.chunk_manifest_ref),
    ]
    for i, ref in enumerate(a3.candidate_extraction_refs, start=1):
        refs.append(LineageRef(f"candidate_extraction_{i:04d}", ref))
    refs.extend(
        [
            LineageRef("candidate_entity_index", entity_map.candidate_entity_index_ref),
            LineageRef("reconciliation_decision_set", entity_map.reconciliation_decision_set_ref),
            LineageRef("canonical_character_registry", entity_map.canonical_character_registry_ref),
            LineageRef("canonical_location_registry", entity_map.canonical_location_registry_ref),
            LineageRef("unresolved_entity_set", entity_map.unresolved_entity_set_ref),
            LineageRef("entity_map", entity_map_ref),
        ]
    )
    return ValidationReport(validated_refs=tuple(refs), findings=())


def _require_a4_validation_report(
    store: FileArtifactStore,
    *,
    artifact_id: str,
    revision: int,
    expected_report: ValidationReport,
) -> ArtifactRef:
    """Verify the exact matching PASS A4 ValidationReport; fail closed on any
    missing / malformed / mismatched / non-PASS report."""
    try:
        ref = store.get(VALIDATION_REPORT_ARTIFACT_TYPE, artifact_id, revision).ref
        report = load_validation_report(store, ref)
        if report != expected_report:
            raise StoryIntegrityError(
                "A4 ValidationReport does not match the exact expected "
                "deterministic validation result"
            )
        if report.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "current EntityMap is backed by a non-PASS A4 ValidationReport"
            )
        return ref
    except StoryIntegrityError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(
            f"matching A4 ValidationReport is missing or invalid for "
            f"{artifact_id} revision {revision}"
        ) from exc


# ---------------------------------------------------------------------------
# Shared fail-closed verification (publication revalidation + CURRENT verify)
# ---------------------------------------------------------------------------


def validate_a4_semantic_identity_binding(
    *,
    decisions: tuple[ReconciliationDecision, ...],
    semantic_identity: A4SemanticIdentity,
) -> None:
    """Fail closed unless every LLM decision binds to the A4 semantic identity.

    For every LLM decision requires, simultaneously:
      * ``method == "llm"`` with a ``generation_provenance``;
      * ``prompt_id`` / ``prompt_version`` equal the semantic identity's;
      * the provenance identity (semantic profile id/hash, prompt id/version
        content hash, output schema id/version/hash) matches the semantic
        identity exactly;
      * the provenance ``request_hash`` is one of the semantic request hashes;
      * ``decision_id`` equals the deterministically recomputed A4C decision id
        (the reason code is derived from the decision itself, never trusted
        from a persisted field).

    Additionally requires exact request-hash coverage: the ordered first
    occurrence of the LLM decisions' request hashes (in canonical pair order)
    equals ``semantic_identity.semantic_request_hashes``. With zero semantic
    pairs there are no LLM decisions and ``semantic_request_hashes == ()``.
    """
    llm_decisions = sorted(
        (d for d in decisions if d.method == "llm"),
        key=lambda d: (d.left_candidate_ref, d.right_candidate_ref),
    )
    request_hashes = set(semantic_identity.semantic_request_hashes)
    for decision in llm_decisions:
        provenance = decision.generation_provenance
        if provenance is None:
            raise StoryIntegrityError(
                f"LLM decision {decision.decision_id!r} has no generation provenance"
            )
        if (
            decision.prompt_id != semantic_identity.prompt_id
            or decision.prompt_version != semantic_identity.prompt_version
        ):
            raise StoryIntegrityError(
                f"LLM decision {decision.decision_id!r} prompt identity "
                f"{decision.prompt_id!r}/{decision.prompt_version!r} does not "
                f"match the semantic identity"
            )
        if (
            provenance.semantic_profile_id != semantic_identity.semantic_profile_id
            or provenance.semantic_profile_hash != semantic_identity.semantic_profile_hash
            or provenance.prompt_id != semantic_identity.prompt_id
            or provenance.prompt_version != semantic_identity.prompt_version
            or provenance.prompt_content_hash != semantic_identity.prompt_content_hash
            or provenance.output_schema_id != semantic_identity.output_schema_id
            or provenance.output_schema_version != semantic_identity.output_schema_version
            or provenance.output_schema_hash != semantic_identity.output_schema_hash
        ):
            raise StoryIntegrityError(
                f"LLM decision {decision.decision_id!r} generation provenance "
                f"identity does not match the semantic identity"
            )
        if provenance.request_hash not in request_hashes:
            raise StoryIntegrityError(
                f"LLM decision {decision.decision_id!r} request_hash "
                f"{provenance.request_hash!r} is not among the semantic request hashes"
            )
        expected_id = compute_llm_decision_id(
            left_ref=decision.left_candidate_ref,
            right_ref=decision.right_candidate_ref,
            decision=decision.decision,
            method="llm",
            reason_code=llm_reason_code(decision.decision),
            reason_zh=decision.reason_zh,
            evidence_refs=decision.evidence_refs,
            prompt_id=decision.prompt_id,
            prompt_version=decision.prompt_version,
            request_hash=provenance.request_hash,
        )
        if decision.decision_id != expected_id:
            raise StoryIntegrityError(
                f"LLM decision {decision.decision_id!r} does not match the "
                f"deterministic decision-id authority (expected {expected_id!r})"
            )
    seen: list[str] = []
    for decision in llm_decisions:
        request_hash = decision.generation_provenance.request_hash  # type: ignore[union-attr]
        if request_hash not in seen:
            seen.append(request_hash)
    if tuple(seen) != semantic_identity.semantic_request_hashes:
        raise StoryIntegrityError(
            "semantic request hashes are not exactly covered by the LLM decisions "
            f"(expected {semantic_identity.semantic_request_hashes!r}, "
            f"got {tuple(seen)!r})"
        )


def _validate_a3_input_index_binding(
    candidate_index: CandidateEntityIndex, a3_input: A3InputIdentity
) -> None:
    """Fail closed unless the candidate index binds to the A3 input identity.

    Requires, simultaneously:
      * the A3 input ``candidate_extraction_refs`` are unique;
      * every candidate's ``candidate_extraction_ref`` is a member of the A3
        input refs;
      * the candidate extraction source order is consistent with the A3 input
        strict source order (non-decreasing A3 ordinal in index source order).
    """
    seen: set[ArtifactRef] = set()
    for ref in a3_input.candidate_extraction_refs:
        if ref in seen:
            raise StoryIntegrityError(
                f"duplicate candidate_extraction_ref in A3 input identity: {ref!r}"
            )
        seen.add(ref)
    a3_ordinals = {ref: i for i, ref in enumerate(a3_input.candidate_extraction_refs)}
    prev_ordinal = -1
    for entry in candidate_index.entries:
        ext_ref = entry.candidate_extraction_ref
        if ext_ref not in a3_ordinals:
            raise StoryIntegrityError(
                f"candidate {entry.candidate_ref!r} references extraction "
                f"{ext_ref!r} not present in the A3 input identity"
            )
        ordinal = a3_ordinals[ext_ref]
        if ordinal < prev_ordinal:
            raise StoryIntegrityError(
                f"candidate {entry.candidate_ref!r} extraction ref {ext_ref!r} "
                f"breaks the A3 input strict source order"
            )
        prev_ordinal = ordinal


def _validate_profile_binding(
    semantic_identity: A4SemanticIdentity,
    profile: EntityReconciliationProfile | None,
    profile_id: str | None,
) -> None:
    """Bind the semantic identity to the reconciliation profile.

    Full binding (profile object) verifies the id AND hash; profile-id-only
    binding (reuse) verifies the id only.
    """
    if profile is not None:
        if semantic_identity.reconciliation_profile_id != profile.profile_id:
            raise StoryIntegrityError(
                "semantic identity reconciliation profile id does not match the "
                "requested reconciliation profile"
            )
        if semantic_identity.reconciliation_profile_hash != profile.profile_hash:
            raise StoryIntegrityError(
                "semantic identity reconciliation profile hash does not match the "
                "requested reconciliation profile"
            )
    elif profile_id is not None:
        if semantic_identity.reconciliation_profile_id != profile_id:
            raise StoryIntegrityError(
                "semantic identity reconciliation profile id does not match the "
                "requested reconciliation profile"
            )


def _verify_finalization_bundle(
    *,
    index: CandidateEntityIndex,
    decision_set: ReconciliationDecisionSet,
    char_registry: CanonicalCharacterRegistry,
    loc_registry: CanonicalLocationRegistry,
    unresolved_set: UnresolvedEntitySet,
    entity_map_entries: tuple,
    a3_input: A3InputIdentity,
    semantic_identity: A4SemanticIdentity,
    profile: EntityReconciliationProfile | None = None,
    profile_id: str | None = None,
) -> None:
    """Independently revalidate a persisted (or in-memory) A4 finalization.

    This is the single source of the fail-closed A4D verification, reused by
    both :meth:`ReconciliationPersistenceService.publish_validated` (independent
    revalidation before persistence) and
    :meth:`ReconciliationPersistenceService._verify_current_entity_map` (CURRENT
    verification). It:

      * gates on the frozen ``source_order_key`` shape / strict source order;
      * gates on unique decision ids;
      * replans the index and requires the replanned ``plan_hash`` to equal the
        semantic identity's ``plan_hash``;
      * validates every decision against the replanned pair plans (deterministic
        decisions must equal the replanned authority; semantic decisions must
        use method ``llm``);
      * exact-compares the persisted outputs against the graph-derived
        expectation (single-source derivation authority);
      * re-runs the deterministic graph/coverage validation;
      * binds the LLM decisions to the semantic identity;
      * binds the candidate index to the A3 input identity;
      * binds the semantic identity to the reconciliation profile.

    Raises :class:`ReconciliationFinalizationError` on blocking findings and
    :class:`StoryIntegrityError` on any other structural violation.
    """
    decisions = decision_set.decisions

    # Structural gates: frozen source order + unique decision ids. A structurally
    # invalid index / decision set fails closed with StoryIntegrityError.
    try:
        validate_candidate_index_source_order(index)
    except ReconciliationPlanningError as exc:
        raise StoryIntegrityError(
            f"A4 candidate index has an invalid source_order_key: {exc}"
        ) from exc
    seen_ids: set[str] = set()
    for decision in decisions:
        if decision.decision_id in seen_ids:
            raise StoryIntegrityError(
                f"duplicate decision_id in decision set: {decision.decision_id!r}"
            )
        seen_ids.add(decision.decision_id)

    # Replan the index and require plan_hash parity.
    replanned = plan_candidate_index_v1(index)
    if replanned.plan_hash != semantic_identity.plan_hash:
        raise StoryIntegrityError(
            f"replanned plan_hash {replanned.plan_hash!r} does not match the "
            f"semantic identity plan_hash {semantic_identity.plan_hash!r}"
        )
    replanned_plans = {
        (p.left_candidate_ref, p.right_candidate_ref): p for p in replanned.pair_plans
    }
    replanned_det = {
        (d.left_candidate_ref, d.right_candidate_ref): d for d in replanned.decisions
    }

    # Validate every decision against the replanned pair plans.
    for decision in decisions:
        key = (decision.left_candidate_ref, decision.right_candidate_ref)
        plan = replanned_plans.get(key)
        if plan is None:
            raise StoryIntegrityError(
                f"decision for pair {key!r} is not present in the replanned pair plans"
            )
        if plan.state in (PAIR_STATE_AUTO_SAME, PAIR_STATE_MUST_NOT_MERGE):
            expected = replanned_det.get(key)
            if decision != expected:
                raise StoryIntegrityError(
                    f"deterministic decision for pair {key!r} does not match the "
                    f"replanned planning authority"
                )
        elif plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            if decision.method != "llm":
                raise StoryIntegrityError(
                    f"semantic decision for pair {key!r} must use method 'llm', "
                    f"got {decision.method!r}"
                )
        else:
            raise StoryIntegrityError(f"unknown pair state {plan.state!r} for pair {key!r}")

    # Exact-compare the persisted outputs against the graph-derived expectation.
    derived = derive_reconciliation_outputs(index, decision_set)
    if (
        derived.canonical_character_registry != char_registry
        or derived.canonical_location_registry != loc_registry
        or derived.unresolved_entity_set != unresolved_set
        or derived.entity_map_entries != entity_map_entries
    ):
        raise StoryIntegrityError(
            "persisted A4 outputs do not exactly match the graph-derived expectation"
        )

    # Re-run the deterministic graph/coverage validation.
    graph = build_identity_graph(index.entries, decisions)
    findings = validate_finalization(
        candidate_index=index,
        decision_set=decision_set,
        pair_plans=replanned.pair_plans,
        graph=graph,
        canonical_character_registry=char_registry,
        canonical_location_registry=loc_registry,
        unresolved_entity_set=unresolved_set,
        entity_map_entries=entity_map_entries,
    )
    if any(f.severity is ValidationSeverity.BLOCKING for f in findings):
        raise ReconciliationFinalizationError(
            "A4 finalization does not re-validate cleanly; not current-eligible",
            findings=tuple(findings),
        )

    # Bind the LLM decisions to the semantic identity.
    validate_a4_semantic_identity_binding(
        decisions=decisions, semantic_identity=semantic_identity
    )
    # Bind the candidate index to the A3 input identity.
    _validate_a3_input_index_binding(index, a3_input)
    # Bind the semantic identity to the reconciliation profile.
    _validate_profile_binding(semantic_identity, profile, profile_id)


# ---------------------------------------------------------------------------
# Pointer helper
# ---------------------------------------------------------------------------


def _current_pointer(
    pointer_store: FilePointerStore, pointer_id: str
) -> tuple[ArtifactRef | None, ArtifactRef | None]:
    try:
        pointer_ref = pointer_store.resolve_current_pointer_ref(pointer_id)
        pointer = pointer_store.resolve_current(pointer_id)
        return pointer_ref, pointer.target_ref
    except PointerNotFoundError:
        return None, None


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationPublication:
    """Outcome of an A4D reconciliation persistence / reuse operation.

    Carries the exact published / reused A4 artifact refs (the five outputs,
    the entity map, the PASS validation report) plus the CURRENT pointer ref
    and whether the operation was a reuse. A4E uses it for the orchestration
    summary.
    """

    candidate_entity_index_ref: ArtifactRef
    reconciliation_decision_set_ref: ArtifactRef
    canonical_character_registry_ref: ArtifactRef
    canonical_location_registry_ref: ArtifactRef
    unresolved_entity_set_ref: ArtifactRef
    entity_map_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    reused: bool


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ReconciliationPersistenceService:
    """Narrow deterministic A4D service: persistence, reuse, and publication.

    A4E drives the two-phase flow: call :meth:`try_reuse_current` before the
    provider call (zero provider calls on a current-eligible exact identity
    match), and only on a miss call the provider, finalize, then
    :meth:`publish_validated` with the in-memory finalization result. The
    service never invokes an LLM, never renders prompts, and never regenerates.
    """

    def __init__(self, store: FileArtifactStore, pointers: FilePointerStore) -> None:
        self.store = store
        self.pointers = pointers

    # -- shared helpers -----------------------------------------------------

    def _check_current_logical_target(self, current_ref: ArtifactRef, logical_id: str) -> None:
        if (
            current_ref.artifact_type != ENTITY_MAP_ARTIFACT_TYPE
            or current_ref.artifact_id != logical_id
        ):
            raise StoryIntegrityError(
                "CURRENT pointer targets a different logical EntityMap"
            )

    def _verify_current_entity_map(
        self,
        *,
        base: str,
        logical_entity_map_id: str,
        pointer_id: str,
        reconciliation_profile_id: str,
        current_pointer_ref: ArtifactRef | None,
        current_entity_map_ref: ArtifactRef,
    ) -> EntityMap:
        """Fully verify the exact CURRENT EntityMap; fail closed on corruption.

        The shared :func:`_verify_finalization_bundle` verifier is reused here so
        publication revalidation and CURRENT verification enforce byte-identical
        checks (plan_hash parity, decision validation against the replanned pair
        plans, graph-derived output parity, semantic / A3 / profile binding).
        """
        self._check_current_logical_target(current_entity_map_ref, logical_entity_map_id)
        entity_map = load_entity_map(
            self.store, current_entity_map_ref, expected_artifact_id=logical_entity_map_id
        )
        run_revision = current_entity_map_ref.revision

        index = load_candidate_entity_index(
            self.store,
            entity_map.candidate_entity_index_ref,
            expected_artifact_id=candidate_entity_index_artifact_id(base),
        )
        decision_set = load_reconciliation_decision_set(
            self.store,
            entity_map.reconciliation_decision_set_ref,
            expected_artifact_id=reconciliation_decision_set_artifact_id(base),
        )
        char_registry = load_canonical_character_registry(
            self.store,
            entity_map.canonical_character_registry_ref,
            expected_artifact_id=canonical_character_registry_artifact_id(base),
        )
        loc_registry = load_canonical_location_registry(
            self.store,
            entity_map.canonical_location_registry_ref,
            expected_artifact_id=canonical_location_registry_artifact_id(base),
        )
        unresolved_set = load_unresolved_entity_set(
            self.store,
            entity_map.unresolved_entity_set_ref,
            expected_artifact_id=unresolved_entity_set_artifact_id(base),
        )

        for ref in (
            entity_map.candidate_entity_index_ref,
            entity_map.reconciliation_decision_set_ref,
            entity_map.canonical_character_registry_ref,
            entity_map.canonical_location_registry_ref,
            entity_map.unresolved_entity_set_ref,
        ):
            if ref.revision != run_revision:
                raise StoryIntegrityError(
                    "A4 output artifacts do not share one run revision with the EntityMap"
                )

        _verify_finalization_bundle(
            index=index,
            decision_set=decision_set,
            char_registry=char_registry,
            loc_registry=loc_registry,
            unresolved_set=unresolved_set,
            entity_map_entries=entity_map.entries,
            a3_input=entity_map.a3_input,
            semantic_identity=entity_map.semantic_identity,
            profile_id=reconciliation_profile_id,
        )

        expected_report = build_a4_validation_report(entity_map, current_entity_map_ref)
        _require_a4_validation_report(
            self.store,
            artifact_id=a4_validation_artifact_id(base),
            revision=run_revision,
            expected_report=expected_report,
        )

        if self.pointers.resolve_current_pointer_ref(pointer_id) != current_pointer_ref:
            raise StoryPersistenceError(
                "A4 CURRENT pointer changed during verification"
            )
        return entity_map

    # -- pre-generation reuse ----------------------------------------------

    def try_reuse_current(
        self,
        *,
        project_id: str,
        document_id: str,
        reconciliation_profile_id: str,
        a3_input: A3InputIdentity,
        semantic_identity: A4SemanticIdentity,
    ) -> ReconciliationPublication | None:
        """Pre-generation current-only reuse check.

        Returns a ``reused`` :class:`ReconciliationPublication` when the exact
        CURRENT EntityMap is current-eligible and its (A3 input, A4 semantic
        identity) exactly matches the requested one; returns ``None`` on a
        normal cache miss (no CURRENT, or a valid CURRENT with a different
        identity); raises (fail closed) when the CURRENT is corrupt, missing
        its exact PASS report, non-PASS, noncanonical, or targets a different
        logical EntityMap.
        """
        base = a4_base_artifact_id(project_id, document_id, reconciliation_profile_id)
        pointer_id = a4_pointer_id(project_id, document_id, reconciliation_profile_id)
        logical_entity_map_id = entity_map_artifact_id(base)
        current_pointer_ref, current_entity_map_ref = _current_pointer(
            self.pointers, pointer_id
        )
        if current_entity_map_ref is None:
            return None
        entity_map = self._verify_current_entity_map(
            base=base,
            logical_entity_map_id=logical_entity_map_id,
            pointer_id=pointer_id,
            reconciliation_profile_id=reconciliation_profile_id,
            current_pointer_ref=current_pointer_ref,
            current_entity_map_ref=current_entity_map_ref,
        )
        if entity_map.a3_input != a3_input or entity_map.semantic_identity != semantic_identity:
            return None
        assert current_pointer_ref is not None
        return ReconciliationPublication(
            candidate_entity_index_ref=entity_map.candidate_entity_index_ref,
            reconciliation_decision_set_ref=entity_map.reconciliation_decision_set_ref,
            canonical_character_registry_ref=entity_map.canonical_character_registry_ref,
            canonical_location_registry_ref=entity_map.canonical_location_registry_ref,
            unresolved_entity_set_ref=entity_map.unresolved_entity_set_ref,
            entity_map_ref=current_entity_map_ref,
            validation_report_ref=_require_a4_validation_report(
                self.store,
                artifact_id=a4_validation_artifact_id(base),
                revision=current_entity_map_ref.revision,
                expected_report=build_a4_validation_report(entity_map, current_entity_map_ref),
            ),
            current_pointer_ref=current_pointer_ref,
            reused=True,
        )

    # -- post-generation publish -------------------------------------------

    def publish_validated(
        self,
        *,
        project_id: str,
        document_id: str,
        reconciliation_profile: EntityReconciliationProfile,
        semantic_identity: A4SemanticIdentity,
        a3_input: A3InputIdentity,
        finalization_result: ReconciliationFinalizationResult,
    ) -> ReconciliationPublication:
        """Post-generation publish: validate (zero blocking findings), persist an
        immutable run revision, publish the exact matching PASS A4
        ValidationReport, and compare-and-set move CURRENT.

        A non-publishable finalization (any blocking finding) raises
        :class:`ReconciliationFinalizationError` before any artifact is written
        and never replaces a valid CURRENT.
        """
        if finalization_result.has_blocking_findings:
            raise ReconciliationFinalizationError(
                "A4 finalization has blocking findings; cannot publish",
                findings=finalization_result.findings,
            )

        profile_id = reconciliation_profile.profile_id
        base = a4_base_artifact_id(project_id, document_id, profile_id)
        pointer_id = a4_pointer_id(project_id, document_id, profile_id)
        logical_entity_map_id = entity_map_artifact_id(base)

        # Independent revalidation of the in-memory finalization BEFORE any
        # artifact is written: replan parity, graph-derived output parity,
        # graph/coverage, semantic / A3 / profile binding. This is the shared
        # verifier, so a new publication and a CURRENT verify enforce identical
        # checks (Blocker 4).
        _verify_finalization_bundle(
            index=finalization_result.candidate_index,
            decision_set=finalization_result.decision_set,
            char_registry=finalization_result.canonical_character_registry,
            loc_registry=finalization_result.canonical_location_registry,
            unresolved_set=finalization_result.unresolved_entity_set,
            entity_map_entries=finalization_result.entity_map_entries,
            a3_input=a3_input,
            semantic_identity=semantic_identity,
            profile=reconciliation_profile,
        )

        # Re-read/re-verify CURRENT at the actual publication boundary (the
        # post-provider race). A corrupt / wrong-logical-target CURRENT fails
        # closed; a valid same-identity CURRENT is reused.
        current_pointer_ref, current_entity_map_ref = _current_pointer(
            self.pointers, pointer_id
        )
        if current_entity_map_ref is not None:
            current_entity_map = self._verify_current_entity_map(
                base=base,
                logical_entity_map_id=logical_entity_map_id,
                pointer_id=pointer_id,
                reconciliation_profile_id=profile_id,
                current_pointer_ref=current_pointer_ref,
                current_entity_map_ref=current_entity_map_ref,
            )
            if (
                current_entity_map.a3_input == a3_input
                and current_entity_map.semantic_identity == semantic_identity
            ):
                assert current_pointer_ref is not None
                return ReconciliationPublication(
                    candidate_entity_index_ref=current_entity_map.candidate_entity_index_ref,
                    reconciliation_decision_set_ref=current_entity_map.reconciliation_decision_set_ref,
                    canonical_character_registry_ref=current_entity_map.canonical_character_registry_ref,
                    canonical_location_registry_ref=current_entity_map.canonical_location_registry_ref,
                    unresolved_entity_set_ref=current_entity_map.unresolved_entity_set_ref,
                    entity_map_ref=current_entity_map_ref,
                    validation_report_ref=_require_a4_validation_report(
                        self.store,
                        artifact_id=a4_validation_artifact_id(base),
                        revision=current_entity_map_ref.revision,
                        expected_report=build_a4_validation_report(
                            current_entity_map, current_entity_map_ref
                        ),
                    ),
                    current_pointer_ref=current_pointer_ref,
                    reused=True,
                )

        # No current (or a different identity): persist a new run revision.
        revision = next_a4_revision(
            self.store, base=base, current_entity_map_ref=current_entity_map_ref
        )
        index_ref = persist_candidate_entity_index(
            self.store,
            finalization_result.candidate_index,
            artifact_id=candidate_entity_index_artifact_id(base),
            revision=revision,
        )
        decision_ref = persist_reconciliation_decision_set(
            self.store,
            finalization_result.decision_set,
            artifact_id=reconciliation_decision_set_artifact_id(base),
            revision=revision,
        )
        char_ref = persist_canonical_character_registry(
            self.store,
            finalization_result.canonical_character_registry,
            artifact_id=canonical_character_registry_artifact_id(base),
            revision=revision,
        )
        loc_ref = persist_canonical_location_registry(
            self.store,
            finalization_result.canonical_location_registry,
            artifact_id=canonical_location_registry_artifact_id(base),
            revision=revision,
        )
        unresolved_ref = persist_unresolved_entity_set(
            self.store,
            finalization_result.unresolved_entity_set,
            artifact_id=unresolved_entity_set_artifact_id(base),
            revision=revision,
        )

        entity_map = EntityMap(
            schema_version=ENTITY_MAP_SCHEMA_VERSION,
            entries=finalization_result.entity_map_entries,
            candidate_entity_index_ref=index_ref,
            reconciliation_decision_set_ref=decision_ref,
            canonical_character_registry_ref=char_ref,
            canonical_location_registry_ref=loc_ref,
            unresolved_entity_set_ref=unresolved_ref,
            a3_input=a3_input,
            semantic_identity=semantic_identity,
        )
        _validate_schema(entity_map.to_dict(), "entity-map.schema.json", "EntityMap")
        entity_map_ref = persist_entity_map(
            self.store,
            entity_map,
            artifact_id=logical_entity_map_id,
            revision=revision,
        )

        report = build_a4_validation_report(entity_map, entity_map_ref)
        if report.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "A4 ValidationReport is not PASS; EntityMap cannot become current"
            )
        report_ref = persist_validation_report(
            self.store,
            report,
            artifact_id=a4_validation_artifact_id(base),
            revision=revision,
        )
        if (
            load_validation_report(self.store, report_ref).summary.result
            is not ValidationResult.PASS
        ):
            raise StoryIntegrityError("persisted A4 ValidationReport is not PASS")

        try:
            pointer_ref = self.pointers.compare_and_set(
                pointer_id=pointer_id,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=current_pointer_ref,
                target_ref=entity_map_ref,
            )
        except Exception as exc:  # noqa: BLE001
            raise StoryPersistenceError(
                "failed to publish A4 CURRENT pointer; immutable artifacts "
                "remain historical"
            ) from exc

        return ReconciliationPublication(
            candidate_entity_index_ref=index_ref,
            reconciliation_decision_set_ref=decision_ref,
            canonical_character_registry_ref=char_ref,
            canonical_location_registry_ref=loc_ref,
            unresolved_entity_set_ref=unresolved_ref,
            entity_map_ref=entity_map_ref,
            validation_report_ref=report_ref,
            current_pointer_ref=pointer_ref,
            reused=False,
        )


__all__ = [
    "CANDIDATE_ENTITY_INDEX_ARTIFACT_TYPE",
    "RECONCILIATION_DECISION_SET_ARTIFACT_TYPE",
    "CANONICAL_CHARACTER_REGISTRY_ARTIFACT_TYPE",
    "CANONICAL_LOCATION_REGISTRY_ARTIFACT_TYPE",
    "UNRESOLVED_ENTITY_SET_ARTIFACT_TYPE",
    "ENTITY_MAP_ARTIFACT_TYPE",
    "ReconciliationPublication",
    "ReconciliationPersistenceService",
    "a4_base_artifact_id",
    "a4_pointer_id",
    "a4_validation_artifact_id",
    "build_a4_validation_report",
    "candidate_entity_index_artifact_id",
    "validate_a4_semantic_identity_binding",
    "canonical_character_registry_artifact_id",
    "canonical_location_registry_artifact_id",
    "entity_map_artifact_id",
    "load_candidate_entity_index",
    "load_canonical_character_registry",
    "load_canonical_location_registry",
    "load_entity_map",
    "load_reconciliation_decision_set",
    "load_unresolved_entity_set",
    "next_a4_revision",
    "persist_candidate_entity_index",
    "persist_canonical_character_registry",
    "persist_canonical_location_registry",
    "persist_entity_map",
    "persist_reconciliation_decision_set",
    "persist_unresolved_entity_set",
    "reconciliation_decision_set_artifact_id",
    "unresolved_entity_set_artifact_id",
]
