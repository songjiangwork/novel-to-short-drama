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
    LineageRef,
    ValidationReport,
    ValidationResult,
    load_validation_report,
)
from short_drama.io import load_json
from short_drama.paths import SCHEMAS_DIR

from .consolidation import (
    CanonicalEventSet,
    CanonicalFactSet,
    CanonicalRelationshipSet,
    ConsolidationCandidateIndex,
    ConsolidationDecisionSet,
    ConsolidationManifest,
    StoryConflictSet,
)
from .errors import StoryIntegrityError, StoryPersistenceError
from .reconciliation_persistence import ENTITY_MAP_ARTIFACT_TYPE


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
    "build_a5_validation_report",
    "canonical_event_set_artifact_id",
    "canonical_fact_set_artifact_id",
    "canonical_relationship_set_artifact_id",
    "consolidation_candidate_index_artifact_id",
    "consolidation_decision_set_artifact_id",
    "consolidation_manifest_artifact_id",
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
