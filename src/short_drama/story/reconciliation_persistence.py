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

from .errors import StoryIntegrityError, StoryPersistenceError
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
    ReconciliationDecisionSet,
    UnresolvedEntitySet,
)
from .reconciliation_finalization import (
    ReconciliationFinalizationError,
    ReconciliationFinalizationResult,
    build_identity_graph,
)
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
    """Outcome of an A4D reconciliation persistence / reuse operation."""

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
        current_pointer_ref: ArtifactRef | None,
        current_entity_map_ref: ArtifactRef,
    ) -> EntityMap:
        """Fully verify the exact CURRENT EntityMap; fail closed on corruption."""
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

        # Re-run the deterministic graph/coverage validation over the persisted
        # artifacts (pair plans are not persisted, so pass None).
        graph = build_identity_graph(index.entries, decision_set.decisions)
        findings = validate_finalization(
            candidate_index=index,
            decision_set=decision_set,
            graph=graph,
            canonical_character_registry=char_registry,
            canonical_location_registry=loc_registry,
            unresolved_entity_set=unresolved_set,
            entity_map_entries=entity_map.entries,
            pair_plans=None,
        )
        if any(f.severity is ValidationSeverity.BLOCKING for f in findings):
            raise StoryIntegrityError(
                "persisted A4 finalization does not re-validate cleanly; not current-eligible"
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
            current_pointer_ref=current_pointer_ref,
            current_entity_map_ref=current_entity_map_ref,
        )
        if entity_map.a3_input != a3_input or entity_map.semantic_identity != semantic_identity:
            return None
        assert current_pointer_ref is not None
        return ReconciliationPublication(
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
                current_pointer_ref=current_pointer_ref,
                current_entity_map_ref=current_entity_map_ref,
            )
            if (
                current_entity_map.a3_input == a3_input
                and current_entity_map.semantic_identity == semantic_identity
            ):
                assert current_pointer_ref is not None
                return ReconciliationPublication(
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
