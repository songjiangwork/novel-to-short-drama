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
    ConsolidationDecisionSet,
    ConsolidationManifest,
    ConsolidationProfile,
    OutputSchemaAssetIdentity,
    PromptAssetIdentity,
    StoryConflictSet,
)
from .consolidation_finalization import A5FinalizationResult
from .consolidation_planning import ConsolidationInputSnapshot, ConsolidationPlanningResult
from .consolidation_semantic import (
    EventSemanticPreparation,
    FactSemanticPreparation,
    RelationshipSemanticPreparation,
)
from .errors import (
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

    The three preparations must agree on the consolidation profile id and the
    deterministic plan hash; otherwise the identity is undefined (fail closed).
    The prompt / output-schema asset identities are collected in domain order
    (fact, event, relationship). The semantic request hashes are the
    domain-ordered concatenation of the three preparations' request hashes
    (fact, then event, then relationship); it may be empty when no domain has
    a semantic block.
    """
    fact_profile = fact_preparation.consolidation_profile
    event_profile = event_preparation.consolidation_profile
    rel_profile = relationship_preparation.consolidation_profile
    if not (
        fact_profile.profile_id
        == event_profile.profile_id
        == rel_profile.profile_id
    ):
        raise StoryIntegrityError(
            "the three domain semantic preparations disagree on the "
            "consolidation profile id"
        )
    fact_plan = fact_preparation.planning_result.plan_hash
    event_plan = event_preparation.planning_result.plan_hash
    rel_plan = relationship_preparation.planning_result.plan_hash
    if not (fact_plan == event_plan == rel_plan):
        raise StoryIntegrityError(
            "the three domain semantic preparations disagree on the plan hash"
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
        plan_hash=fact_plan,
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
    the PASS validation report) plus the CURRENT pointer target ref. A5F2 never
    reuses (``reused`` is always ``False``); current-only exact reuse is A5F3.
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


class ConsolidationPersistenceService:
    """The A5F2 publication-boundary service.

    Mirrors :meth:`ReconciliationPersistenceService.publish_validated` (A4D):
    it persists the A5 run (six leaves + manifest + deterministic PASS report)
    at one shared run revision, re-checks upstream A4 CURRENT stability, and
    publishes the A5 CURRENT via a compare-and-set (``PointerKind.CURRENT``).
    It performs NO provider calls and never reuses (A5F3 adds reuse).
    """

    def __init__(self, store: FileArtifactStore, pointers: FilePointerStore) -> None:
        self._store = store
        self._pointers = pointers

    # -- shared helpers -----------------------------------------------------

    def _current_a5_pointer(
        self, pointer_id: str
    ) -> tuple[ArtifactRef | None, ArtifactRef | None]:
        """Read the A5 CURRENT leniently: ``(pointer_ref, target_ref)`` or
        ``(None, None)`` when absent."""
        try:
            pointer_ref = self._pointers.resolve_current_pointer_ref(pointer_id)
            pointer = self._pointers.resolve_current(pointer_id)
            return pointer_ref, pointer.target_ref
        except PointerNotFoundError:
            return None, None

    @staticmethod
    def _verify_semantic_identity_binding(
        *,
        consolidation_profile: ConsolidationProfile,
        semantic_identity: A5SemanticIdentity,
        planning_result: ConsolidationPlanningResult,
    ) -> None:
        """Bind the semantic identity to the publication profile + planning result.

        The semantic identity must pin the exact consolidation profile (id +
        content hash) and the exact deterministic plan hash the run was built
        from; a mismatch is a structural A5 integrity failure.
        """
        if semantic_identity.consolidation_profile_id != consolidation_profile.profile_id:
            raise StoryIntegrityError(
                "semantic_identity.consolidation_profile_id does not match the "
                "publication consolidation profile"
            )
        if semantic_identity.consolidation_profile_hash != consolidation_profile.content_hash():
            raise StoryIntegrityError(
                "semantic_identity.consolidation_profile_hash does not match the "
                "publication consolidation profile content hash"
            )
        if semantic_identity.plan_hash != planning_result.plan_hash:
            raise StoryIntegrityError(
                "semantic_identity.plan_hash does not match the publication "
                "planning result plan hash"
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

    # -- publication --------------------------------------------------------

    def publish_validated(
        self,
        *,
        project_id: str,
        document_id: str,
        consolidation_profile: ConsolidationProfile,
        semantic_identity: A5SemanticIdentity,
        finalization_result: A5FinalizationResult,
        decision_set: ConsolidationDecisionSet,
    ) -> ConsolidationPublication:
        """Publish a validated A5 state and compare-and-set the A5 CURRENT.

        Order: (1) bind the semantic identity to the profile + planning result,
        (2) allocate the shared run revision (skip orphans), (3) persist the six
        leaves, (4) build + persist the manifest, (5) build + persist the
        deterministic PASS A5 ValidationReport, (6) re-check upstream A4 CURRENT
        stability, (7) CAS the A5 CURRENT. A failure at any step leaves the
        existing A5 CURRENT untouched (any new artifacts remain historical).
        """
        if not isinstance(consolidation_profile, ConsolidationProfile):
            raise StoryIntegrityError(
                "consolidation_profile must be a ConsolidationProfile"
            )
        if not isinstance(semantic_identity, A5SemanticIdentity):
            raise StoryIntegrityError(
                "semantic_identity must be an A5SemanticIdentity"
            )
        if not isinstance(finalization_result, A5FinalizationResult):
            raise StoryIntegrityError(
                "finalization_result must be an A5FinalizationResult"
            )
        if not isinstance(decision_set, ConsolidationDecisionSet):
            raise StoryIntegrityError(
                "decision_set must be a ConsolidationDecisionSet"
            )

        planning_result = finalization_result.planning_result
        snapshot = planning_result.snapshot
        profile_id = consolidation_profile.profile_id
        base = a5_base_artifact_id(project_id, document_id, profile_id)
        pointer_id = a5_pointer_id(project_id, document_id, profile_id)
        reconciliation_profile_id = (
            snapshot.entity_map.semantic_identity.reconciliation_profile_id
        )

        # (1) Bind the semantic identity to the profile + planning result.
        self._verify_semantic_identity_binding(
            consolidation_profile=consolidation_profile,
            semantic_identity=semantic_identity,
            planning_result=planning_result,
        )

        # (2) Read the A5 CURRENT leniently and allocate the shared run revision.
        current_pointer_ref, current_manifest_ref = self._current_a5_pointer(pointer_id)
        revision = next_a5_revision(
            self._store, base=base, current_manifest_ref=current_manifest_ref
        )

        # (3) Persist the six leaves (shared run revision).
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

        # (4) Build + persist the manifest.
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
            coverage_summary=planning_result.coverage,
        )
        manifest_ref = persist_consolidation_manifest(
            self._store,
            manifest,
            artifact_id=consolidation_manifest_artifact_id(base),
            revision=revision,
        )

        # (5) Build + persist the deterministic PASS A5 ValidationReport.
        report = build_a5_validation_report(manifest, manifest_ref)
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

        # (6) Re-check upstream A4 CURRENT stability (before the CAS).
        self._require_a4_upstream_stable(
            project_id=project_id,
            document_id=document_id,
            reconciliation_profile_id=reconciliation_profile_id,
            snapshot=snapshot,
        )

        # (7) CAS the A5 CURRENT (CURRENT kind; no authority ref).
        try:
            self._pointers.compare_and_set(
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
            current_pointer_ref=manifest_ref,
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
