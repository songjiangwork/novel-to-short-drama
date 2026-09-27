"""v1.2 A5F1 — consolidation persistence primitives.

This module is the A5F1 *persistence primitives* layer. It sits on top of the
A5A domain models (``consolidation.py``) and the existing Foundation artifact
/ validation authorities. It owns:

  * deterministic A5 artifact identity (base + six leaf outputs + manifest +
    A5 validation report + the future CURRENT pointer id);
  * one shared run revision across all eight A5 artifact slots, with
    orphan-aware next-revision allocation;
  * immutable A5 persistence + fail-closed typed loaders (tracked JSON Schema
    gates, canonical ``to_dict`` parity, ArtifactRef hash integrity);
  * the exact deterministic A5 PASS ``ValidationReport`` and its exact-matching
    verification;
  * a private manifest reference verifier (root identity, leaf logical ids /
    types, shared run revision, and leaf resolvability) that A5F2 reuses.

It is deliberately a *primitives-only* slice: it performs NO provider calls,
NO pointer mutation, NO CURRENT publication, and NO reuse. The deterministic
``a5_pointer_id`` helper is frozen here, but nothing in this module resolves or
writes the pointer. A5F2 adds the publication-boundary semantic verification,
CURRENT CAS, and upstream-stability re-check; A5F3 adds current-only exact
reuse.

A5 reuses the existing ``FileArtifactStore`` / ``ImmutableArtifactEnvelope`` /
``content_hash`` / canonical-JSON / ``ValidationReport`` authorities from the
Foundation rather than inventing a second artifact store, pointer system,
content-hash format, ValidationReport model, or serialization system.
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
    load_validation_report,
    persist_validation_report,
)
from short_drama.io import load_json
from short_drama.paths import SCHEMAS_DIR

from .consolidation import (
    A5SemanticIdentity,
    A5UpstreamIdentity,
    CanonicalEventSet,
    CanonicalFactSet,
    CanonicalRelationshipSet,
    ConsolidationCandidateIndex,
    ConsolidationCoverageSummary,
    ConsolidationDecisionSet,
    ConsolidationManifest,
    ConsolidationProfile,
    OutputSchemaAssetIdentity,
    PromptAssetIdentity,
    StoryConflictSet,
)
from .consolidation_finalization import A5FinalizationResult, finalize_consolidation
from .consolidation_planning import ConsolidationInputSnapshot, ConsolidationPlanningResult
from .consolidation_semantic import (
    EventSemanticPreparation,
    EventSemanticResolutionResult,
    FactSemanticPreparation,
    FactSemanticResolutionResult,
    RelationshipSemanticPreparation,
    RelationshipSemanticResolutionResult,
)
from .errors import (
    ConsolidationCurrentMissingError,
    ConsolidationUpstreamUnstableError,
    StoryIntegrityError,
    StoryPersistenceError,
)
from .reconciliation_persistence import (
    ENTITY_MAP_ARTIFACT_TYPE,
    ReconciliationPersistenceService,
)


# ---------------------------------------------------------------------------
# Artifact types (stable A5 artifact types)
# ---------------------------------------------------------------------------

CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE = "consolidation_candidate_index"
CONSOLIDATION_DECISION_SET_ARTIFACT_TYPE = "consolidation_decision_set"
CANONICAL_FACT_SET_ARTIFACT_TYPE = "canonical_fact_set"
CANONICAL_EVENT_SET_ARTIFACT_TYPE = "canonical_event_set"
CANONICAL_RELATIONSHIP_SET_ARTIFACT_TYPE = "canonical_relationship_set"
STORY_CONFLICT_SET_ARTIFACT_TYPE = "story_conflict_set"
CONSOLIDATION_MANIFEST_ARTIFACT_TYPE = "consolidation_manifest"


# ---------------------------------------------------------------------------
# Schema versions (all A5A contracts are version 1)
# ---------------------------------------------------------------------------

CONSOLIDATION_CANDIDATE_INDEX_SCHEMA_VERSION = 1
CONSOLIDATION_DECISION_SET_SCHEMA_VERSION = 1
CANONICAL_FACT_SET_SCHEMA_VERSION = 1
CANONICAL_EVENT_SET_SCHEMA_VERSION = 1
CANONICAL_RELATIONSHIP_SET_SCHEMA_VERSION = 1
STORY_CONFLICT_SET_SCHEMA_VERSION = 1
CONSOLIDATION_MANIFEST_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Artifact identity (frozen A5F1 contract)
# ---------------------------------------------------------------------------


def a5_base_artifact_id(
    project_id: str, document_id: str, consolidation_profile_id: str
) -> str:
    """Deterministic A5 base identity.

    ``<project_id>.<document_id>.a5.<consolidation_profile_id>``
    """
    return f"{project_id}.{document_id}.a5.{consolidation_profile_id}"


def consolidation_candidate_index_artifact_id(base: str) -> str:
    return f"{base}.candidate-index"


def consolidation_decision_set_artifact_id(base: str) -> str:
    return f"{base}.decisions"


def canonical_fact_set_artifact_id(base: str) -> str:
    return f"{base}.facts"


def canonical_event_set_artifact_id(base: str) -> str:
    return f"{base}.events"


def canonical_relationship_set_artifact_id(base: str) -> str:
    return f"{base}.relationships"


def story_conflict_set_artifact_id(base: str) -> str:
    return f"{base}.conflicts"


def consolidation_manifest_artifact_id(base: str) -> str:
    return f"{base}.manifest"


def a5_validation_artifact_id(base: str) -> str:
    return f"{base}.manifest.a5-validation"


def a5_pointer_id(
    project_id: str, document_id: str, consolidation_profile_id: str
) -> str:
    """Deterministic A5 CURRENT-pointer identity (frozen; not written here).

    ``<project_id>.a5.<document_id>.<consolidation_profile_id>``

    A5F1 defines the pointer identity only. It does NOT resolve or write the
    pointer (CURRENT publication is A5F2).
    """
    return f"{project_id}.a5.{document_id}.{consolidation_profile_id}"


def _a5_revision_slots(base: str) -> tuple[tuple[str, str], ...]:
    """The eight A5 artifact slots that share one run revision.

    A complete A5 publication revision uses the same integer revision for all
    six leaf artifacts, the manifest, and the A5 ValidationReport.
    """
    return (
        (
            CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
            consolidation_candidate_index_artifact_id(base),
        ),
        (
            CONSOLIDATION_DECISION_SET_ARTIFACT_TYPE,
            consolidation_decision_set_artifact_id(base),
        ),
        (CANONICAL_FACT_SET_ARTIFACT_TYPE, canonical_fact_set_artifact_id(base)),
        (CANONICAL_EVENT_SET_ARTIFACT_TYPE, canonical_event_set_artifact_id(base)),
        (
            CANONICAL_RELATIONSHIP_SET_ARTIFACT_TYPE,
            canonical_relationship_set_artifact_id(base),
        ),
        (STORY_CONFLICT_SET_ARTIFACT_TYPE, story_conflict_set_artifact_id(base)),
        (CONSOLIDATION_MANIFEST_ARTIFACT_TYPE, consolidation_manifest_artifact_id(base)),
        (VALIDATION_REPORT_ARTIFACT_TYPE, a5_validation_artifact_id(base)),
    )


# ---------------------------------------------------------------------------
# Revision allocation (shared run revision, skip orphans)
# ---------------------------------------------------------------------------


def next_a5_revision(
    store: FileArtifactStore,
    *,
    base: str,
    current_manifest_ref: ArtifactRef | None,
) -> int:
    """Allocate the next shared run revision across all eight A5 artifact slots.

    A run revision must be free for every A5 artifact identity (the six leaf
    artifacts, the manifest, and the A5 ValidationReport). Collisions with
    already-existing historical revisions (including orphans left by a failed
    publication) are skipped; an existing revision is never overwritten.

    When no current manifest ref is supplied the allocation starts at revision
    1; when one is supplied it starts at its revision + 1. A candidate revision
    is free only when none of the eight slots occupies it.
    """
    start = 1 if current_manifest_ref is None else current_manifest_ref.revision + 1
    revision = max(1, start)
    while True:
        occupied = False
        for artifact_type, artifact_id in _a5_revision_slots(base):
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
) -> ImmutableArtifactEnvelope:
    try:
        return ImmutableArtifactEnvelope.create(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            revision=revision,
            schema_version=schema_version,
            payload=payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(f"failed to build {label} envelope: {exc}") from exc


def persist_consolidation_candidate_index(
    store: FileArtifactStore,
    index: ConsolidationCandidateIndex,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    if not isinstance(index, ConsolidationCandidateIndex):
        raise StoryIntegrityError("index must be a ConsolidationCandidateIndex")
    envelope = _envelope(
        CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CONSOLIDATION_CANDIDATE_INDEX_SCHEMA_VERSION,
        index.to_dict(),
        "ConsolidationCandidateIndex",
    )
    return _put(store, envelope, "ConsolidationCandidateIndex")


def persist_consolidation_decision_set(
    store: FileArtifactStore,
    decision_set: ConsolidationDecisionSet,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    if not isinstance(decision_set, ConsolidationDecisionSet):
        raise StoryIntegrityError("decision_set must be a ConsolidationDecisionSet")
    envelope = _envelope(
        CONSOLIDATION_DECISION_SET_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CONSOLIDATION_DECISION_SET_SCHEMA_VERSION,
        decision_set.to_dict(),
        "ConsolidationDecisionSet",
    )
    return _put(store, envelope, "ConsolidationDecisionSet")


def persist_canonical_fact_set(
    store: FileArtifactStore,
    fact_set: CanonicalFactSet,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    if not isinstance(fact_set, CanonicalFactSet):
        raise StoryIntegrityError("fact_set must be a CanonicalFactSet")
    envelope = _envelope(
        CANONICAL_FACT_SET_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CANONICAL_FACT_SET_SCHEMA_VERSION,
        fact_set.to_dict(),
        "CanonicalFactSet",
    )
    return _put(store, envelope, "CanonicalFactSet")


def persist_canonical_event_set(
    store: FileArtifactStore,
    event_set: CanonicalEventSet,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    if not isinstance(event_set, CanonicalEventSet):
        raise StoryIntegrityError("event_set must be a CanonicalEventSet")
    envelope = _envelope(
        CANONICAL_EVENT_SET_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CANONICAL_EVENT_SET_SCHEMA_VERSION,
        event_set.to_dict(),
        "CanonicalEventSet",
    )
    return _put(store, envelope, "CanonicalEventSet")


def persist_canonical_relationship_set(
    store: FileArtifactStore,
    relationship_set: CanonicalRelationshipSet,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    if not isinstance(relationship_set, CanonicalRelationshipSet):
        raise StoryIntegrityError(
            "relationship_set must be a CanonicalRelationshipSet"
        )
    envelope = _envelope(
        CANONICAL_RELATIONSHIP_SET_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CANONICAL_RELATIONSHIP_SET_SCHEMA_VERSION,
        relationship_set.to_dict(),
        "CanonicalRelationshipSet",
    )
    return _put(store, envelope, "CanonicalRelationshipSet")


def persist_story_conflict_set(
    store: FileArtifactStore,
    conflict_set: StoryConflictSet,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    if not isinstance(conflict_set, StoryConflictSet):
        raise StoryIntegrityError("conflict_set must be a StoryConflictSet")
    envelope = _envelope(
        STORY_CONFLICT_SET_ARTIFACT_TYPE,
        artifact_id,
        revision,
        STORY_CONFLICT_SET_SCHEMA_VERSION,
        conflict_set.to_dict(),
        "StoryConflictSet",
    )
    return _put(store, envelope, "StoryConflictSet")


def persist_consolidation_manifest(
    store: FileArtifactStore,
    manifest: ConsolidationManifest,
    *,
    artifact_id: str,
    revision: int,
) -> ArtifactRef:
    if not isinstance(manifest, ConsolidationManifest):
        raise StoryIntegrityError("manifest must be a ConsolidationManifest")
    envelope = _envelope(
        CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        artifact_id,
        revision,
        CONSOLIDATION_MANIFEST_SCHEMA_VERSION,
        manifest.to_dict(),
        "ConsolidationManifest",
    )
    return _put(store, envelope, "ConsolidationManifest")


def _put(
    store: FileArtifactStore, envelope: ImmutableArtifactEnvelope, label: str
) -> ArtifactRef:
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


def load_consolidation_candidate_index(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> ConsolidationCandidateIndex:
    return _load_typed(
        store,
        ref,
        artifact_type=CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
        schema_version=CONSOLIDATION_CANDIDATE_INDEX_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="consolidation-candidate-index.schema.json",
        label="ConsolidationCandidateIndex",
        loader=ConsolidationCandidateIndex.from_dict,
    )


def load_consolidation_decision_set(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> ConsolidationDecisionSet:
    return _load_typed(
        store,
        ref,
        artifact_type=CONSOLIDATION_DECISION_SET_ARTIFACT_TYPE,
        schema_version=CONSOLIDATION_DECISION_SET_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="consolidation-decision-set.schema.json",
        label="ConsolidationDecisionSet",
        loader=ConsolidationDecisionSet.from_dict,
    )


def load_canonical_fact_set(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> CanonicalFactSet:
    return _load_typed(
        store,
        ref,
        artifact_type=CANONICAL_FACT_SET_ARTIFACT_TYPE,
        schema_version=CANONICAL_FACT_SET_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="canonical-fact-set.schema.json",
        label="CanonicalFactSet",
        loader=CanonicalFactSet.from_dict,
    )


def load_canonical_event_set(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> CanonicalEventSet:
    return _load_typed(
        store,
        ref,
        artifact_type=CANONICAL_EVENT_SET_ARTIFACT_TYPE,
        schema_version=CANONICAL_EVENT_SET_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="canonical-event-set.schema.json",
        label="CanonicalEventSet",
        loader=CanonicalEventSet.from_dict,
    )


def load_canonical_relationship_set(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> CanonicalRelationshipSet:
    return _load_typed(
        store,
        ref,
        artifact_type=CANONICAL_RELATIONSHIP_SET_ARTIFACT_TYPE,
        schema_version=CANONICAL_RELATIONSHIP_SET_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="canonical-relationship-set.schema.json",
        label="CanonicalRelationshipSet",
        loader=CanonicalRelationshipSet.from_dict,
    )


def load_story_conflict_set(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> StoryConflictSet:
    return _load_typed(
        store,
        ref,
        artifact_type=STORY_CONFLICT_SET_ARTIFACT_TYPE,
        schema_version=STORY_CONFLICT_SET_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="story-conflict-set.schema.json",
        label="StoryConflictSet",
        loader=StoryConflictSet.from_dict,
    )


def load_consolidation_manifest(
    store: FileArtifactStore, ref: ArtifactRef, *, expected_artifact_id: str
) -> ConsolidationManifest:
    return _load_typed(
        store,
        ref,
        artifact_type=CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        schema_version=CONSOLIDATION_MANIFEST_SCHEMA_VERSION,
        expected_artifact_id=expected_artifact_id,
        schema_filename="consolidation-manifest.schema.json",
        label="ConsolidationManifest",
        loader=ConsolidationManifest.from_dict,
    )


# ---------------------------------------------------------------------------
# Manifest reference verification (A5F2 reuses)
# ---------------------------------------------------------------------------


def _verify_consolidation_manifest(
    store: FileArtifactStore,
    *,
    base: str,
    manifest: ConsolidationManifest,
    manifest_ref: ArtifactRef,
) -> None:
    """Fail-closed structural verification of a manifest's references.

    Requires, simultaneously:
      * the manifest ref is the exact logical A5 manifest
        (``artifact_type == consolidation_manifest`` and
        ``artifact_id == <base>.manifest``);
      * the manifest's OWN logical identity agrees with ``base``: the base
        reconstructed from the manifest's ``project_id`` / ``document_id`` /
        ``semantic_identity.consolidation_profile_id`` equals ``base`` (checked
        before any leaf logical identity is trusted);
      * the upstream ``entity_map_ref`` is structurally an ``entity_map`` ref
        (A5F1 checks the type only; A4 CURRENT stability publication belongs
        to A5F2 and is deliberately NOT performed here);
      * every A5 leaf ref has the correct artifact type, the exact expected
        logical artifact id, and shares the manifest's run revision;
      * every leaf ref resolves via its typed loader (missing / corrupt /
        wrong-hash / wrong-logical-target leaves fail closed).

    A5F1 does NOT require CURRENT and performs NO provider calls.
    """
    if manifest_ref.artifact_type != CONSOLIDATION_MANIFEST_ARTIFACT_TYPE:
        raise StoryIntegrityError(
            "ConsolidationManifest ref has the wrong artifact_type"
        )
    if manifest_ref.artifact_id != consolidation_manifest_artifact_id(base):
        raise StoryIntegrityError(
            "ConsolidationManifest artifact_id does not match its logical identity"
        )
    # Bind the manifest's own content identity to ``base`` BEFORE trusting any
    # leaf logical identity: the base must be exactly the deterministic base
    # reconstructed from the manifest's own identity fields. This is checked by
    # reconstructing ``base`` (never by splitting the base string on dots).
    expected_base = a5_base_artifact_id(
        manifest.project_id,
        manifest.document_id,
        manifest.semantic_identity.consolidation_profile_id,
    )
    if expected_base != base:
        raise StoryIntegrityError(
            "ConsolidationManifest content identity does not match the A5 base"
        )
    if manifest.entity_map_ref.artifact_type != ENTITY_MAP_ARTIFACT_TYPE:
        raise StoryIntegrityError(
            "ConsolidationManifest entity_map_ref must be an entity_map ref"
        )

    run_revision = manifest_ref.revision
    leaf_specs = (
        (
            "candidate index",
            CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE,
            consolidation_candidate_index_artifact_id(base),
            manifest.consolidation_candidate_index_ref,
            load_consolidation_candidate_index,
        ),
        (
            "decision set",
            CONSOLIDATION_DECISION_SET_ARTIFACT_TYPE,
            consolidation_decision_set_artifact_id(base),
            manifest.consolidation_decision_set_ref,
            load_consolidation_decision_set,
        ),
        (
            "fact set",
            CANONICAL_FACT_SET_ARTIFACT_TYPE,
            canonical_fact_set_artifact_id(base),
            manifest.canonical_fact_set_ref,
            load_canonical_fact_set,
        ),
        (
            "event set",
            CANONICAL_EVENT_SET_ARTIFACT_TYPE,
            canonical_event_set_artifact_id(base),
            manifest.canonical_event_set_ref,
            load_canonical_event_set,
        ),
        (
            "relationship set",
            CANONICAL_RELATIONSHIP_SET_ARTIFACT_TYPE,
            canonical_relationship_set_artifact_id(base),
            manifest.canonical_relationship_set_ref,
            load_canonical_relationship_set,
        ),
        (
            "conflict set",
            STORY_CONFLICT_SET_ARTIFACT_TYPE,
            story_conflict_set_artifact_id(base),
            manifest.story_conflict_set_ref,
            load_story_conflict_set,
        ),
    )
    for label, artifact_type, expected_id, ref, loader in leaf_specs:
        if ref.artifact_type != artifact_type:
            raise StoryIntegrityError(
                f"manifest {label} ref has the wrong artifact_type"
            )
        if ref.artifact_id != expected_id:
            raise StoryIntegrityError(
                f"manifest {label} artifact_id does not match its logical identity"
            )
        if ref.revision != run_revision:
            raise StoryIntegrityError(
                f"manifest {label} does not share the manifest run revision"
            )
        # Resolvability: the leaf must load cleanly (missing / corrupt /
        # wrong-hash / wrong-logical-target all fail closed).
        loader(store, ref, expected_artifact_id=expected_id)


# ---------------------------------------------------------------------------
# A5 ValidationReport (frozen deterministic contract)
# ---------------------------------------------------------------------------


def build_a5_validation_report(
    manifest: ConsolidationManifest, manifest_ref: ArtifactRef
) -> ValidationReport:
    """The exact deterministic PASS A5 ValidationReport for a manifest.

    Lineage (roles deterministic): the exact upstream A3 refs (source document,
    chunk manifest, and each candidate extraction by deterministic ordinal
    ``candidate_extraction_NNNN``) plus the upstream ``entity_map`` and all six
    A5 outputs (candidate index, decision set, canonical fact / event /
    relationship sets, story conflict set) and the manifest itself. The
    successful report has ``findings = ()``. The first A3 refs come from
    ``manifest.upstream_identity.a3_input``; A5 does not independently search
    for upstream artifacts.
    """
    a3 = manifest.upstream_identity.a3_input
    refs: list[LineageRef] = [
        LineageRef("source_document", a3.source_document_ref),
        LineageRef("chunk_manifest", a3.chunk_manifest_ref),
    ]
    for i, ref in enumerate(a3.candidate_extraction_refs, start=1):
        refs.append(LineageRef(f"candidate_extraction_{i:04d}", ref))
    refs.extend(
        [
            LineageRef("entity_map", manifest.entity_map_ref),
            LineageRef(
                "consolidation_candidate_index",
                manifest.consolidation_candidate_index_ref,
            ),
            LineageRef(
                "consolidation_decision_set",
                manifest.consolidation_decision_set_ref,
            ),
            LineageRef("canonical_fact_set", manifest.canonical_fact_set_ref),
            LineageRef("canonical_event_set", manifest.canonical_event_set_ref),
            LineageRef(
                "canonical_relationship_set",
                manifest.canonical_relationship_set_ref,
            ),
            LineageRef("story_conflict_set", manifest.story_conflict_set_ref),
            LineageRef("consolidation_manifest", manifest_ref),
        ]
    )
    return ValidationReport(validated_refs=tuple(refs), findings=())


def _require_a5_validation_report(
    store: FileArtifactStore,
    *,
    artifact_id: str,
    revision: int,
    expected_report: ValidationReport,
) -> ArtifactRef:
    """Verify the exact matching PASS A5 ValidationReport; fail closed on any
    missing / malformed / mismatched / non-PASS report.

    There is no fallback historical search: the report must exist at the exact
    ``artifact_id`` / ``revision``, be a valid canonical ValidationReport, equal
    ``expected_report`` exactly, and be PASS.
    """
    try:
        ref = store.get(VALIDATION_REPORT_ARTIFACT_TYPE, artifact_id, revision).ref
        report = load_validation_report(store, ref)
        if report != expected_report:
            raise StoryIntegrityError(
                "A5 ValidationReport does not match the exact expected "
                "deterministic validation result"
            )
        if report.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "manifest is backed by a non-PASS A5 ValidationReport"
            )
        return ref
    except StoryIntegrityError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(
            f"matching A5 ValidationReport is missing or invalid for "
            f"{artifact_id} revision {revision}"
        ) from exc


# ---------------------------------------------------------------------------
# A5F2 — validated publication + upstream A4 stability + CAS CURRENT
# ---------------------------------------------------------------------------


def build_a5_semantic_identity(
    fact_preparation: FactSemanticPreparation,
    event_preparation: EventSemanticPreparation,
    relationship_preparation: RelationshipSemanticPreparation,
) -> A5SemanticIdentity:
    """Build the backend-neutral A5 semantic identity from the three domain
    semantic preparations.

    The three preparations must agree on the EXACT consolidation profile (full
    value equality, not merely the profile id) and the EXACT
    :class:`ConsolidationPlanningResult` (full value equality, not merely the
    plan hash); otherwise the identity is undefined (fail closed). This is the
    frozen A5F2 publication-authority gate: a same-id / same-plan-hash but
    otherwise-different profile or planning object is a structural failure.
    The prompt / output-schema asset identities are collected in domain order
    (fact, event, relationship). The semantic request hashes are the
    domain-ordered concatenation of the three preparations' request hashes
    (fact, then event, then relationship); it may be empty when no domain has
    a semantic block.
    """
    fact_profile = fact_preparation.consolidation_profile
    event_profile = event_preparation.consolidation_profile
    rel_profile = relationship_preparation.consolidation_profile
    if not (fact_profile == event_profile == rel_profile):
        raise StoryIntegrityError(
            "the three domain semantic preparations disagree on the "
            "consolidation profile"
        )
    fact_plan = fact_preparation.planning_result
    event_plan = event_preparation.planning_result
    rel_plan = relationship_preparation.planning_result
    if not (fact_plan == event_plan == rel_plan):
        raise StoryIntegrityError(
            "the three domain semantic preparations disagree on the "
            "planning result"
        )
    return A5SemanticIdentity(
        consolidation_profile_id=fact_profile.profile_id,
        consolidation_profile_hash=fact_profile.content_hash(),
        fact_semantic_profile_id=fact_preparation.semantic_profile.profile_id,
        fact_semantic_profile_hash=fact_preparation.semantic_profile.semantic_profile_hash,
        event_semantic_profile_id=event_preparation.semantic_profile.profile_id,
        event_semantic_profile_hash=event_preparation.semantic_profile.semantic_profile_hash,
        relationship_semantic_profile_id=relationship_preparation.semantic_profile.profile_id,
        relationship_semantic_profile_hash=relationship_preparation.semantic_profile.semantic_profile_hash,
        prompt_identities=(
            PromptAssetIdentity(
                prompt_id=fact_preparation.prompt_id,
                prompt_version=fact_preparation.prompt_version,
                prompt_content_hash=fact_preparation.prompt_content_hash,
            ),
            PromptAssetIdentity(
                prompt_id=event_preparation.prompt_id,
                prompt_version=event_preparation.prompt_version,
                prompt_content_hash=event_preparation.prompt_content_hash,
            ),
            PromptAssetIdentity(
                prompt_id=relationship_preparation.prompt_id,
                prompt_version=relationship_preparation.prompt_version,
                prompt_content_hash=relationship_preparation.prompt_content_hash,
            ),
        ),
        output_schema_identities=(
            OutputSchemaAssetIdentity(
                schema_id=fact_preparation.output_schema_id,
                schema_version=fact_preparation.output_schema_version,
                schema_hash=fact_preparation.output_schema_hash,
            ),
            OutputSchemaAssetIdentity(
                schema_id=event_preparation.output_schema_id,
                schema_version=event_preparation.output_schema_version,
                schema_hash=event_preparation.output_schema_hash,
            ),
            OutputSchemaAssetIdentity(
                schema_id=relationship_preparation.output_schema_id,
                schema_version=relationship_preparation.output_schema_version,
                schema_hash=relationship_preparation.output_schema_hash,
            ),
        ),
        plan_hash=fact_plan.plan_hash,
        semantic_request_hashes=(
            fact_preparation.semantic_request_hashes
            + event_preparation.semantic_request_hashes
            + relationship_preparation.semantic_request_hashes
        ),
    )


@dataclass(frozen=True, slots=True)
class ConsolidationPublication:
    """Outcome of an A5F2 validated publication.

    Carries the exact published A5 artifact refs (the six leaves, the manifest,
    the PASS validation report), the immutable current_pointer artifact ref
    (``current_pointer_ref``), and the pointer target ref (which is the
    ``consolidation_manifest_ref``). A5F2 never reuses (``reused`` is always
    ``False``); current-only exact reuse is A5F3.
    """

    consolidation_candidate_index_ref: ArtifactRef
    consolidation_decision_set_ref: ArtifactRef
    canonical_fact_set_ref: ArtifactRef
    canonical_event_set_ref: ArtifactRef
    canonical_relationship_set_ref: ArtifactRef
    story_conflict_set_ref: ArtifactRef
    consolidation_manifest_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    reused: bool


@dataclass(frozen=True, slots=True)
class ValidatedConsolidationCurrent:
    """A fully-verified, current-eligible A5 CURRENT manifest resolved read-only.

    A5F3 (and any downstream A5 consumer) must consume A5 strictly through its
    CURRENT pointer. This is the return value of
    :meth:`ConsolidationPersistenceService.require_current_validated`: the exact
    CURRENT :class:`ConsolidationManifest` (already verified end-to-end: logical
    target, structural bundle, leaf resolvability, and the exact PASS A5
    ValidationReport), its artifact ref, the exact PASS validation-report ref,
    and the CURRENT pointer ref.

    This is a read-only downstream seam: it performs no writes and no CURRENT
    mutation. It raises :class:`ConsolidationCurrentMissingError` when no
    current-eligible A5 CURRENT exists (a structural A5 failure, not a cache
    miss).
    """

    manifest: ConsolidationManifest
    manifest_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    current_pointer_ref: ArtifactRef


class ConsolidationPersistenceService:
    """The A5F2/A5F3 consolidation publication and reuse service.

    Mirrors :meth:`ReconciliationPersistenceService.publish_validated` (A4D):
    it persists the A5 run (six leaves + manifest + deterministic PASS report)
    at one shared run revision, re-checks upstream A4 CURRENT stability, and
    publishes the A5 CURRENT via a compare-and-set (``PointerKind.CURRENT``).
    A5F3 adds a read-only, current-only exact-reuse seam.  Neither path makes
    provider calls.
    """

    def __init__(self, store: FileArtifactStore, pointers: FilePointerStore) -> None:
        self._store = store
        self._pointers = pointers

    # -- shared helpers -----------------------------------------------------

    def _current_a5_pointer(
        self, pointer_id: str
    ) -> tuple[ArtifactRef | None, ArtifactRef | None]:
        """Read the A5 CURRENT leniently: ``(pointer_ref, target_ref)`` or
        ``(None, None)`` when absent.

        A5's frozen CURRENT contract requires ``PointerKind.CURRENT`` (never
        ``CURRENT_APPROVED``). An existing pointer with a different kind is a
        structural control-plane failure and fails closed.
        """
        try:
            pointer_ref = self._pointers.resolve_current_pointer_ref(pointer_id)
            pointer = self._pointers.resolve_current(pointer_id)
        except PointerNotFoundError:
            return None, None
        if pointer.pointer_kind is not PointerKind.CURRENT:
            raise StoryIntegrityError(
                "A5 CURRENT pointer must use PointerKind.CURRENT"
            )
        return pointer_ref, pointer.target_ref

    def _verify_current_consolidation(
        self,
        *,
        base: str,
        pointer_id: str,
        current_pointer_ref: ArtifactRef | None,
        current_manifest_ref: ArtifactRef,
    ) -> ConsolidationManifest:
        """Fully verify the exact CURRENT A5 manifest; fail closed on corruption.

        The shared :func:`_verify_consolidation_manifest` structural verifier is
        reused so publication revalidation and CURRENT verification enforce
        byte-identical checks (logical target, base identity, leaf logical
        ids / types, shared run revision, leaf resolvability). In addition the
        exact PASS A5 ValidationReport is required and the CURRENT pointer must
        be stable for the duration of the verification.
        """
        if current_manifest_ref.artifact_type != CONSOLIDATION_MANIFEST_ARTIFACT_TYPE:
            raise StoryIntegrityError(
                "A5 CURRENT pointer targets a different artifact type"
            )
        if current_manifest_ref.artifact_id != consolidation_manifest_artifact_id(base):
            raise StoryIntegrityError(
                "A5 CURRENT pointer targets a different logical A5 manifest"
            )
        manifest = load_consolidation_manifest(
            self._store,
            current_manifest_ref,
            expected_artifact_id=consolidation_manifest_artifact_id(base),
        )
        _verify_consolidation_manifest(
            self._store,
            base=base,
            manifest=manifest,
            manifest_ref=current_manifest_ref,
        )
        expected_report = build_a5_validation_report(manifest, current_manifest_ref)
        _require_a5_validation_report(
            self._store,
            artifact_id=a5_validation_artifact_id(base),
            revision=current_manifest_ref.revision,
            expected_report=expected_report,
        )
        if self._pointers.resolve_current_pointer_ref(pointer_id) != current_pointer_ref:
            raise StoryPersistenceError(
                "A5 CURRENT pointer changed during verification"
            )
        return manifest

    def require_current_validated(
        self,
        *,
        project_id: str,
        document_id: str,
        consolidation_profile_id: str,
    ) -> ValidatedConsolidationCurrent:
        """Resolve the exact current-eligible A5 CURRENT manifest, read-only.

        A5F3 (and any downstream A5 consumer) must consume A5 strictly through
        its CURRENT pointer. This method verifies the exact CURRENT end-to-end
        (byte-identical to the A5F2 publication verifier) and returns it,
        together with the exact PASS validation-report ref and the CURRENT
        pointer ref. It performs no writes and no CURRENT mutation.

        A missing CURRENT is a structural A5 failure and raises
        :class:`ConsolidationCurrentMissingError` (not a normal cache miss).
        """
        base = a5_base_artifact_id(project_id, document_id, consolidation_profile_id)
        pointer_id = a5_pointer_id(project_id, document_id, consolidation_profile_id)
        current_pointer_ref, current_manifest_ref = self._current_a5_pointer(pointer_id)
        if current_manifest_ref is None:
            raise ConsolidationCurrentMissingError(
                f"no current-eligible A5 CURRENT manifest for "
                f"{project_id}/{document_id}/{consolidation_profile_id}; "
                "A5 requires a valid A5 CURRENT (structural failure, not a "
                "cache miss)"
            )
        manifest = self._verify_current_consolidation(
            base=base,
            pointer_id=pointer_id,
            current_pointer_ref=current_pointer_ref,
            current_manifest_ref=current_manifest_ref,
        )
        assert current_pointer_ref is not None
        return ValidatedConsolidationCurrent(
            manifest=manifest,
            manifest_ref=current_manifest_ref,
            validation_report_ref=_require_a5_validation_report(
                self._store,
                artifact_id=a5_validation_artifact_id(base),
                revision=current_manifest_ref.revision,
                expected_report=build_a5_validation_report(
                    manifest, current_manifest_ref
                ),
            ),
            current_pointer_ref=current_pointer_ref,
        )

    @staticmethod
    def _final_coverage_summary(
        *,
        planning_result: ConsolidationPlanningResult,
        fact_resolution: FactSemanticResolutionResult,
        event_resolution: EventSemanticResolutionResult,
        relationship_resolution: RelationshipSemanticResolutionResult,
        finalization_result: A5FinalizationResult,
    ) -> ConsolidationCoverageSummary:
        """Build the exact FINAL A5 coverage summary.

        Derived from the planning candidate index (candidate counts) and the
        A5E finalization result (canonical / conflict counts) -- NOT the A5B
        Phase-A placeholder (which carries zero final counts). The uncertain
        decision count is the number of complete fact / event / relationship
        decisions whose decision is ``uncertain``.
        """
        uncertain_decision_count = sum(
            1
            for decision in (
                *fact_resolution.all_fact_decisions,
                *event_resolution.all_event_decisions,
                *relationship_resolution.all_relationship_decisions,
            )
            if decision.decision == "uncertain"
        )
        return ConsolidationCoverageSummary(
            fact_candidate_count=len(planning_result.index.facts),
            event_candidate_count=len(planning_result.index.events),
            relationship_candidate_count=len(planning_result.index.relationships),
            canonical_fact_count=len(finalization_result.canonical_fact_set.facts),
            canonical_event_count=len(finalization_result.canonical_event_set.events),
            canonical_relationship_count=len(
                finalization_result.canonical_relationship_set.relationships
            ),
            uncertain_decision_count=uncertain_decision_count,
            story_conflict_count=len(finalization_result.story_conflict_set.conflicts),
        )

    def _require_a4_upstream_stable(
        self,
        *,
        project_id: str,
        document_id: str,
        reconciliation_profile_id: str,
        snapshot: ConsolidationInputSnapshot,
    ) -> None:
        """Re-check that the A4 CURRENT is still the exact upstream A4 A5 consumed.

        A5 publishes against the exact A4 EntityMap + A3 input identity. If the
        A4 CURRENT advanced (a different EntityMap ref or a different A3 input)
        between the A5 run start and the publication CAS, the in-memory A5 state
        is stale and the publication fails closed.
        """
        a4_service = ReconciliationPersistenceService(self._store, self._pointers)
        current = a4_service.require_current_validated(
            project_id=project_id,
            document_id=document_id,
            reconciliation_profile_id=reconciliation_profile_id,
        )
        if current.entity_map_ref != snapshot.entity_map_ref:
            raise ConsolidationUpstreamUnstableError(
                "upstream A4 CURRENT EntityMap advanced during the A5 run; "
                "refusing to publish a stale A5 state"
            )
        if current.entity_map.a3_input != snapshot.a3_input:
            raise ConsolidationUpstreamUnstableError(
                "upstream A4 CURRENT A3 input advanced during the A5 run; "
                "refusing to publish a stale A5 state"
            )

    # -- pre-provider current-only reuse ----------------------------------

    def try_reuse_current(
        self,
        *,
        project_id: str,
        document_id: str,
        consolidation_profile: ConsolidationProfile,
        planning_result: ConsolidationPlanningResult,
        fact_preparation: FactSemanticPreparation,
        event_preparation: EventSemanticPreparation,
        relationship_preparation: RelationshipSemanticPreparation,
    ) -> ConsolidationPublication | None:
        """Return the exact verified A5 CURRENT on a pre-provider cache hit.

        This intentionally examines *only* the A5 CURRENT pointer.  A missing
        pointer, or a fully valid CURRENT whose upstream or backend-neutral
        semantic identity differs, is a normal read-only miss.  An existing
        CURRENT is always fully verified before it can be compared: corruption
        is therefore a fail-closed error rather than a cache miss.
        """
        if not isinstance(consolidation_profile, ConsolidationProfile):
            raise StoryIntegrityError(
                "consolidation_profile must be a ConsolidationProfile"
            )
        if not isinstance(planning_result, ConsolidationPlanningResult):
            raise StoryIntegrityError(
                "planning_result must be a ConsolidationPlanningResult"
            )
        for domain, preparation, expected_type in (
            ("fact", fact_preparation, FactSemanticPreparation),
            ("event", event_preparation, EventSemanticPreparation),
            ("relationship", relationship_preparation, RelationshipSemanticPreparation),
        ):
            if not isinstance(preparation, expected_type):
                raise StoryIntegrityError(
                    f"{domain}_preparation must be a {expected_type.__name__}"
                )
            if preparation.planning_result != planning_result:
                raise StoryIntegrityError(
                    f"{domain}_preparation.planning_result does not exactly "
                    "match the reuse planning_result"
                )
            if preparation.consolidation_profile != consolidation_profile:
                raise StoryIntegrityError(
                    f"{domain}_preparation.consolidation_profile does not "
                    "exactly match the reuse consolidation_profile"
                )

        snapshot = planning_result.snapshot
        if snapshot.source_document.project_id != project_id:
            raise StoryIntegrityError(
                "planning_result source_document.project_id does not match the "
                "reuse project_id"
            )
        if snapshot.source_document.document_id != document_id:
            raise StoryIntegrityError(
                "planning_result source_document.document_id does not match the "
                "reuse document_id"
            )

        # The one frozen identity builder is deliberately used here too.  It
        # preserves fact -> event -> relationship request-hash ordering and
        # contains no RuntimeConfig/provider routing fields.
        semantic_identity = build_a5_semantic_identity(
            fact_preparation,
            event_preparation,
            relationship_preparation,
        )
        profile_id = consolidation_profile.profile_id
        base = a5_base_artifact_id(project_id, document_id, profile_id)
        pointer_id = a5_pointer_id(project_id, document_id, profile_id)
        current_pointer_ref, current_manifest_ref = self._current_a5_pointer(pointer_id)
        if current_manifest_ref is None:
            return None

        manifest = self._verify_current_consolidation(
            base=base,
            pointer_id=pointer_id,
            current_pointer_ref=current_pointer_ref,
            current_manifest_ref=current_manifest_ref,
        )
        if (
            manifest.entity_map_ref != snapshot.entity_map_ref
            or manifest.upstream_identity.a3_input != snapshot.a3_input
            or manifest.semantic_identity != semantic_identity
        ):
            return None

        # The identity matched a CURRENT which could have become stale between
        # A5B planning and this read-only hit, so retain A5F2's exact A4
        # stability authority before returning it.
        self._require_a4_upstream_stable(
            project_id=project_id,
            document_id=document_id,
            reconciliation_profile_id=(
                snapshot.entity_map.semantic_identity.reconciliation_profile_id
            ),
            snapshot=snapshot,
        )
        assert current_pointer_ref is not None
        return ConsolidationPublication(
            consolidation_candidate_index_ref=manifest.consolidation_candidate_index_ref,
            consolidation_decision_set_ref=manifest.consolidation_decision_set_ref,
            canonical_fact_set_ref=manifest.canonical_fact_set_ref,
            canonical_event_set_ref=manifest.canonical_event_set_ref,
            canonical_relationship_set_ref=manifest.canonical_relationship_set_ref,
            story_conflict_set_ref=manifest.story_conflict_set_ref,
            consolidation_manifest_ref=current_manifest_ref,
            validation_report_ref=_require_a5_validation_report(
                self._store,
                artifact_id=a5_validation_artifact_id(base),
                revision=current_manifest_ref.revision,
                expected_report=build_a5_validation_report(manifest, current_manifest_ref),
            ),
            current_pointer_ref=current_pointer_ref,
            reused=True,
        )

    # -- publication --------------------------------------------------------

    def publish_validated(
        self,
        *,
        project_id: str,
        document_id: str,
        consolidation_profile: ConsolidationProfile,
        planning_result: ConsolidationPlanningResult,
        fact_resolution: FactSemanticResolutionResult,
        event_resolution: EventSemanticResolutionResult,
        relationship_resolution: RelationshipSemanticResolutionResult,
        finalization_result: A5FinalizationResult,
    ) -> ConsolidationPublication:
        """Publish a validated A5 state and compare-and-set the A5 CURRENT.

        The A5F2 publication boundary. The backend-neutral A5 semantic identity
        and the exact :class:`ConsolidationDecisionSet` are assembled from the
        three domain semantic resolution results (NOT taken from the caller), so
        a caller cannot smuggle in a mismatched publication authority.

        Frozen order (fail closed at any step; the existing A5 CURRENT is left
        untouched and any new artifacts remain historical):

          1. in-memory preflight: type checks, project/document binding, exact
             planning / preparation / profile binding (value equality), and an
             independent A5E re-finalization (the supplied finalization must
             exactly match a fresh :func:`finalize_consolidation`);
          2. build the A5 semantic identity, the decision set, and the exact
             final coverage summary (all in-memory);
          3. read the A5 CURRENT leniently; if it exists, fully verify it;
          4. allocate the shared run revision (skip orphans);
          5. persist the six leaves (shared run revision);
          6. persist the manifest;
          7. reload + verify the persisted manifest;
          8. build + persist + reload + verify the deterministic PASS A5
             ValidationReport;
          9. re-check upstream A4 CURRENT stability;
          10. CAS the A5 CURRENT (``PointerKind.CURRENT``; no authority ref).
        """
        # (1) In-memory preflight.
        if not isinstance(consolidation_profile, ConsolidationProfile):
            raise StoryIntegrityError(
                "consolidation_profile must be a ConsolidationProfile"
            )
        if not isinstance(planning_result, ConsolidationPlanningResult):
            raise StoryIntegrityError(
                "planning_result must be a ConsolidationPlanningResult"
            )
        if not isinstance(fact_resolution, FactSemanticResolutionResult):
            raise StoryIntegrityError(
                "fact_resolution must be a FactSemanticResolutionResult"
            )
        if not isinstance(event_resolution, EventSemanticResolutionResult):
            raise StoryIntegrityError(
                "event_resolution must be an EventSemanticResolutionResult"
            )
        if not isinstance(relationship_resolution, RelationshipSemanticResolutionResult):
            raise StoryIntegrityError(
                "relationship_resolution must be a "
                "RelationshipSemanticResolutionResult"
            )
        if not isinstance(finalization_result, A5FinalizationResult):
            raise StoryIntegrityError(
                "finalization_result must be an A5FinalizationResult"
            )

        snapshot = planning_result.snapshot

        # Project/document binding: the planning must be bound to the requested
        # project / document (not merely to some A4 CURRENT).
        if snapshot.source_document.project_id != project_id:
            raise StoryIntegrityError(
                "planning_result source_document.project_id does not match the "
                "publication project_id"
            )
        if snapshot.source_document.document_id != document_id:
            raise StoryIntegrityError(
                "planning_result source_document.document_id does not match the "
                "publication document_id"
            )

        # Exact planning / preparation / profile binding (value equality): every
        # domain resolution and its preparation, and the finalization, must be
        # built from the EXACT planning result and consolidation profile the
        # publication is publishing.
        for domain, resolution in (
            ("fact", fact_resolution),
            ("event", event_resolution),
            ("relationship", relationship_resolution),
        ):
            if resolution.planning_result != planning_result:
                raise StoryIntegrityError(
                    f"{domain}_resolution.planning_result does not exactly "
                    "match the publication planning_result"
                )
            if resolution.preparation.planning_result != planning_result:
                raise StoryIntegrityError(
                    f"{domain}_resolution.preparation.planning_result does not "
                    "exactly match the publication planning_result"
                )
            if resolution.preparation.consolidation_profile != consolidation_profile:
                raise StoryIntegrityError(
                    f"{domain}_resolution.preparation.consolidation_profile does "
                    "not exactly match the publication consolidation profile"
                )
        if finalization_result.planning_result != planning_result:
            raise StoryIntegrityError(
                "finalization_result.planning_result does not exactly match the "
                "publication planning_result"
            )

        # Independent A5E re-finalization: the supplied finalization must
        # exactly match a fresh finalize_consolidation run (0 new artifacts on
        # a mismatch).
        expected_finalization = finalize_consolidation(
            planning_result,
            fact_resolution,
            event_resolution,
            relationship_resolution,
        )
        if expected_finalization != finalization_result:
            raise StoryIntegrityError(
                "the supplied finalization_result does not exactly match the "
                "independently re-finalized A5E result"
            )

        profile_id = consolidation_profile.profile_id
        base = a5_base_artifact_id(project_id, document_id, profile_id)
        pointer_id = a5_pointer_id(project_id, document_id, profile_id)

        # (2) Build the backend-neutral A5 semantic identity, the decision set,
        # and the exact final coverage summary (all in-memory).
        semantic_identity = build_a5_semantic_identity(
            fact_resolution.preparation,
            event_resolution.preparation,
            relationship_resolution.preparation,
        )
        decision_set = ConsolidationDecisionSet(
            schema_version=1,
            fact_decisions=fact_resolution.all_fact_decisions,
            event_decisions=event_resolution.all_event_decisions,
            relationship_decisions=relationship_resolution.all_relationship_decisions,
        )
        coverage_summary = self._final_coverage_summary(
            planning_result=planning_result,
            fact_resolution=fact_resolution,
            event_resolution=event_resolution,
            relationship_resolution=relationship_resolution,
            finalization_result=finalization_result,
        )

        # (3) Read the A5 CURRENT leniently; if it exists, fully verify it.
        current_pointer_ref, current_manifest_ref = self._current_a5_pointer(pointer_id)
        if current_manifest_ref is not None:
            self._verify_current_consolidation(
                base=base,
                pointer_id=pointer_id,
                current_pointer_ref=current_pointer_ref,
                current_manifest_ref=current_manifest_ref,
            )

        # (4) Allocate the shared run revision (skip orphans).
        revision = next_a5_revision(
            self._store, base=base, current_manifest_ref=current_manifest_ref
        )

        # (5) Persist the six leaves (shared run revision).
        index_ref = persist_consolidation_candidate_index(
            self._store,
            planning_result.index,
            artifact_id=consolidation_candidate_index_artifact_id(base),
            revision=revision,
        )
        decision_ref = persist_consolidation_decision_set(
            self._store,
            decision_set,
            artifact_id=consolidation_decision_set_artifact_id(base),
            revision=revision,
        )
        fact_ref = persist_canonical_fact_set(
            self._store,
            finalization_result.canonical_fact_set,
            artifact_id=canonical_fact_set_artifact_id(base),
            revision=revision,
        )
        event_ref = persist_canonical_event_set(
            self._store,
            finalization_result.canonical_event_set,
            artifact_id=canonical_event_set_artifact_id(base),
            revision=revision,
        )
        relationship_ref = persist_canonical_relationship_set(
            self._store,
            finalization_result.canonical_relationship_set,
            artifact_id=canonical_relationship_set_artifact_id(base),
            revision=revision,
        )
        conflict_ref = persist_story_conflict_set(
            self._store,
            finalization_result.story_conflict_set,
            artifact_id=story_conflict_set_artifact_id(base),
            revision=revision,
        )

        # (6) Build + persist the manifest.
        manifest = ConsolidationManifest(
            schema_version=CONSOLIDATION_MANIFEST_SCHEMA_VERSION,
            project_id=project_id,
            document_id=document_id,
            entity_map_ref=snapshot.entity_map_ref,
            consolidation_candidate_index_ref=index_ref,
            consolidation_decision_set_ref=decision_ref,
            canonical_fact_set_ref=fact_ref,
            canonical_event_set_ref=event_ref,
            canonical_relationship_set_ref=relationship_ref,
            story_conflict_set_ref=conflict_ref,
            semantic_identity=semantic_identity,
            upstream_identity=A5UpstreamIdentity(a3_input=snapshot.a3_input),
            coverage_summary=coverage_summary,
        )
        manifest_ref = persist_consolidation_manifest(
            self._store,
            manifest,
            artifact_id=consolidation_manifest_artifact_id(base),
            revision=revision,
        )

        # (7) Reload + verify the persisted manifest (byte-identical re-check).
        persisted_manifest = load_consolidation_manifest(
            self._store,
            manifest_ref,
            expected_artifact_id=consolidation_manifest_artifact_id(base),
        )
        if persisted_manifest != manifest:
            raise StoryIntegrityError(
                "persisted A5 manifest does not match the in-memory manifest"
            )
        _verify_consolidation_manifest(
            self._store,
            base=base,
            manifest=persisted_manifest,
            manifest_ref=manifest_ref,
        )

        # (8) Build + persist + reload + verify the deterministic PASS report.
        report = build_a5_validation_report(persisted_manifest, manifest_ref)
        if report.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "A5 ValidationReport is not PASS; manifest cannot become current"
            )
        report_ref = persist_validation_report(
            self._store,
            report,
            artifact_id=a5_validation_artifact_id(base),
            revision=revision,
        )
        _require_a5_validation_report(
            self._store,
            artifact_id=a5_validation_artifact_id(base),
            revision=revision,
            expected_report=report,
        )

        # (9) Re-check upstream A4 CURRENT stability (before the CAS).
        reconciliation_profile_id = (
            snapshot.entity_map.semantic_identity.reconciliation_profile_id
        )
        self._require_a4_upstream_stable(
            project_id=project_id,
            document_id=document_id,
            reconciliation_profile_id=reconciliation_profile_id,
            snapshot=snapshot,
        )

        # (10) CAS the A5 CURRENT (CURRENT kind; no authority ref).
        try:
            pointer_ref = self._pointers.compare_and_set(
                pointer_id=pointer_id,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=current_pointer_ref,
                target_ref=manifest_ref,
            )
        except Exception as exc:  # noqa: BLE001
            raise StoryPersistenceError(
                "failed to publish A5 CURRENT pointer; immutable artifacts "
                "remain historical"
            ) from exc

        return ConsolidationPublication(
            consolidation_candidate_index_ref=index_ref,
            consolidation_decision_set_ref=decision_ref,
            canonical_fact_set_ref=fact_ref,
            canonical_event_set_ref=event_ref,
            canonical_relationship_set_ref=relationship_ref,
            story_conflict_set_ref=conflict_ref,
            consolidation_manifest_ref=manifest_ref,
            validation_report_ref=report_ref,
            current_pointer_ref=pointer_ref,
            reused=False,
        )


__all__ = [
    "CONSOLIDATION_CANDIDATE_INDEX_ARTIFACT_TYPE",
    "CONSOLIDATION_DECISION_SET_ARTIFACT_TYPE",
    "CANONICAL_FACT_SET_ARTIFACT_TYPE",
    "CANONICAL_EVENT_SET_ARTIFACT_TYPE",
    "CANONICAL_RELATIONSHIP_SET_ARTIFACT_TYPE",
    "STORY_CONFLICT_SET_ARTIFACT_TYPE",
    "CONSOLIDATION_MANIFEST_ARTIFACT_TYPE",
    "CONSOLIDATION_CANDIDATE_INDEX_SCHEMA_VERSION",
    "CONSOLIDATION_DECISION_SET_SCHEMA_VERSION",
    "CANONICAL_FACT_SET_SCHEMA_VERSION",
    "CANONICAL_EVENT_SET_SCHEMA_VERSION",
    "CANONICAL_RELATIONSHIP_SET_SCHEMA_VERSION",
    "STORY_CONFLICT_SET_SCHEMA_VERSION",
    "CONSOLIDATION_MANIFEST_SCHEMA_VERSION",
    "a5_base_artifact_id",
    "a5_pointer_id",
    "a5_validation_artifact_id",
    "build_a5_semantic_identity",
    "build_a5_validation_report",
    "canonical_event_set_artifact_id",
    "canonical_fact_set_artifact_id",
    "canonical_relationship_set_artifact_id",
    "consolidation_candidate_index_artifact_id",
    "consolidation_decision_set_artifact_id",
    "consolidation_manifest_artifact_id",
    "ConsolidationPersistenceService",
    "ConsolidationPublication",
    "ValidatedConsolidationCurrent",
    "load_canonical_event_set",
    "load_canonical_fact_set",
    "load_canonical_relationship_set",
    "load_consolidation_candidate_index",
    "load_consolidation_decision_set",
    "load_consolidation_manifest",
    "load_story_conflict_set",
    "next_a5_revision",
    "persist_canonical_event_set",
    "persist_canonical_fact_set",
    "persist_canonical_relationship_set",
    "persist_consolidation_candidate_index",
    "persist_consolidation_decision_set",
    "persist_consolidation_manifest",
    "persist_story_conflict_set",
    "story_conflict_set_artifact_id",
]
