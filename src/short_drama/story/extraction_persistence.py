"""v1.2 A3C — CandidateExtraction persistence, ValidationReport, and reuse.

This module is the *persistence / reuse* layer for A3 (chunk extraction). It
sits strictly on top of the merged A3A domain models and A3B semantic
validation. It receives an already-validated (or candidate) payload plus the
exact A1/A2 authorities and the A-I3 ``LLMInvocationProvenance`` and owns:

  * deterministic CandidateExtraction / ValidationReport / CURRENT-pointer
    artifact identity;
  * immutable CandidateExtraction persistence + a fail-closed typed loader;
  * the exact A3 ``ValidationReport`` (lineage + deterministic findings) and
    its compare-and-set publication ordering;
  * current-only semantic reuse: resolve the exact CURRENT extraction,
    re-verify its integrity, re-run the deterministic A3B semantic validation,
    verify the exact matching PASS ``ValidationReport``, guard against the
    pointer changing during verification, and either reuse it or publish a new
    revision under the same logical artifact identity.

It deliberately does NOT:

  * call an LLM or build a semantic request (A3D);
  * retry / regenerate a semantic-invalid payload (A3D);
  * provide a stage CLI or real-Qwen smoke (A3D/A3E);
  * scan historical revisions to resurrect an old semantic identity.

It reuses the existing Foundation primitives (``ArtifactRef``,
``FileArtifactStore``, ``FilePointerStore`` / ``CURRENT``, ``LineageRef``,
``ValidationReport``, ``persist_validation_report`` /
``load_validation_report``) and the A1/A2 persistence helpers rather than
inventing a second infrastructure layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from short_drama.artifacts import (
    ArtifactError,
    ArtifactRef,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
)
from short_drama.foundation import (
    FilePointerStore,
    LineageRef,
    PointerKind,
    PointerNotFoundError,
    VALIDATION_REPORT_ARTIFACT_TYPE,
    ValidationReport,
    ValidationResult,
    load_validation_report,
    persist_validation_report,
)
from short_drama.io import load_json
from short_drama.llm import LLMInvocationProvenance
from short_drama.paths import SCHEMAS_DIR

from .chunking import SOURCE_CHUNK_ARTIFACT_TYPE, SourceChunk
from .errors import StoryIntegrityError, StoryPersistenceError
from .extraction import (
    CANDIDATE_EXTRACTION_ARTIFACT_TYPE,
    CANDIDATE_EXTRACTION_SCHEMA_VERSION,
    CandidateExtraction,
    StoryExtractionProfile,
)
from .extraction_validation import (
    canonicalize_candidate_payload,
    validate_candidate_payload,
)
from .persistence import (
    _next_free_revision,
    load_source_chunk,
    load_source_document,
    source_chunk_artifact_id,
    source_document_artifact_id,
)
from .source import SOURCE_DOCUMENT_ARTIFACT_TYPE, SourceDocument


# ---------------------------------------------------------------------------
# Artifact identity (frozen A-I4 contract, section 20)
# ---------------------------------------------------------------------------


def candidate_extraction_artifact_id(
    project_id: str,
    document_id: str,
    chunk_profile_id: str,
    chunk_id: str,
    extraction_profile_id: str,
) -> str:
    """Deterministic logical CandidateExtraction artifact identity.

    ``<project_id>.<document_id>.<chunk_profile_id>.<chunk_id-lower>.<extraction_profile_id>``
    """
    return (
        f"{project_id}.{document_id}.{chunk_profile_id}"
        f".{chunk_id.lower()}.{extraction_profile_id}"
    )


def candidate_extraction_validation_artifact_id(
    artifact_id: str,
) -> str:
    """Deterministic A3 ValidationReport artifact identity.

    ``<candidate_extraction_artifact_id>.a3-validation``
    """
    return f"{artifact_id}.a3-validation"


def candidate_extraction_pointer_id(
    project_id: str,
    document_id: str,
    chunk_profile_id: str,
    chunk_id: str,
    extraction_profile_id: str,
) -> str:
    """Deterministic A3 CURRENT-pointer identity.

    ``<project_id>.a3.<document_id>.<chunk_profile_id>.<chunk_id-lower>.<extraction_profile_id>``

    A3 has no formal approval gate, so the pointer kind is ``CURRENT`` (never
    ``CURRENT_APPROVED``).
    """
    return (
        f"{project_id}.a3.{document_id}.{chunk_profile_id}"
        f".{chunk_id.lower()}.{extraction_profile_id}"
    )


# ---------------------------------------------------------------------------
# Revision allocation (reuses the A1/A2 ``_next_free_revision`` pattern)
# ---------------------------------------------------------------------------


def next_candidate_extraction_revision(
    store: FileArtifactStore,
    *,
    project_id: str,
    document_id: str,
    chunk_profile_id: str,
    chunk_id: str,
    extraction_profile_id: str,
    current_target_ref: ArtifactRef | None,
) -> int:
    """Allocate the next immutable revision for a CandidateExtraction identity.

    Collisions with already-existing historical revisions are avoided (including
    orphaned revisions left by a failed publication); an existing revision is
    never overwritten.
    """
    start = 1 if current_target_ref is None else current_target_ref.revision + 1
    return _next_free_revision(
        store,
        artifact_type=CANDIDATE_EXTRACTION_ARTIFACT_TYPE,
        artifact_id=candidate_extraction_artifact_id(
            project_id, document_id, chunk_profile_id, chunk_id, extraction_profile_id
        ),
        start=start,
    )


# ---------------------------------------------------------------------------
# Immutable persistence
# ---------------------------------------------------------------------------


def persist_candidate_extraction(
    store: FileArtifactStore,
    extraction: CandidateExtraction,
    *,
    revision: int,
) -> ArtifactRef:
    if not isinstance(extraction, CandidateExtraction):
        raise StoryIntegrityError("extraction must be a CandidateExtraction")
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=CANDIDATE_EXTRACTION_ARTIFACT_TYPE,
        artifact_id=candidate_extraction_artifact_id(
            extraction.project_id,
            extraction.document_id,
            extraction.chunk_profile_id,
            extraction.chunk_id,
            extraction.extraction_profile_id,
        ),
        revision=revision,
        schema_version=CANDIDATE_EXTRACTION_SCHEMA_VERSION,
        payload=extraction.to_dict(),
    )
    try:
        return store.put(envelope)
    except ArtifactError as exc:
        raise StoryPersistenceError(
            f"failed to persist CandidateExtraction: {exc}"
        ) from exc


def load_candidate_extraction(
    store: FileArtifactStore,
    ref: ArtifactRef,
) -> CandidateExtraction:
    """Fail-closed typed loader for a persisted CandidateExtraction.

    Verifies, at minimum (frozen contract section 4): the ref is an
    ``ArtifactRef`` of the correct artifact type; the envelope schema version is
    supported; the payload is an object that loads through
    ``CandidateExtraction.from_dict``; the payload round-trips exactly; the
    artifact ID exactly matches the payload identity; the source-ref artifact
    types are correct; and the extraction identity is internally coherent.
    """
    if (
        not isinstance(ref, ArtifactRef)
        or ref.artifact_type != CANDIDATE_EXTRACTION_ARTIFACT_TYPE
    ):
        raise StoryIntegrityError(
            "CandidateExtraction ref has the wrong artifact_type"
        )
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise StoryIntegrityError(
            f"failed to resolve CandidateExtraction: {ref!r}"
        ) from exc
    if envelope.schema_version != CANDIDATE_EXTRACTION_SCHEMA_VERSION:
        raise StoryIntegrityError(
            "unsupported CandidateExtraction schema_version: "
            f"{envelope.schema_version}; "
            f"supported={CANDIDATE_EXTRACTION_SCHEMA_VERSION}"
        )
    payload = envelope.payload
    if not isinstance(payload, dict):
        raise StoryIntegrityError(
            "persisted CandidateExtraction payload must be an object"
        )
    try:
        extraction = CandidateExtraction.from_dict(payload)
    except Exception as exc:  # noqa: BLE001 - report any typed-model failure
        raise StoryIntegrityError(
            f"invalid persisted CandidateExtraction: {exc}"
        ) from exc

    # The persisted payload must round-trip to the exact typed object.
    if extraction.to_dict() != payload:
        raise StoryIntegrityError(
            "persisted CandidateExtraction payload is not in canonical "
            "semantic form"
        )

    # Artifact ID must exactly match the payload identity.
    expected_id = candidate_extraction_artifact_id(
        extraction.project_id,
        extraction.document_id,
        extraction.chunk_profile_id,
        extraction.chunk_id,
        extraction.extraction_profile_id,
    )
    if ref.artifact_id != expected_id:
        raise StoryIntegrityError(
            "CandidateExtraction artifact_id does not match payload identity"
        )

    # Source refs must carry the correct artifact types and be internally
    # coherent with the extraction's own identity.
    if extraction.source_document_ref.artifact_type != SOURCE_DOCUMENT_ARTIFACT_TYPE:
        raise StoryIntegrityError(
            "CandidateExtraction.source_document_ref has the wrong artifact_type"
        )
    if extraction.source_chunk_ref.artifact_type != SOURCE_CHUNK_ARTIFACT_TYPE:
        raise StoryIntegrityError(
            "CandidateExtraction.source_chunk_ref has the wrong artifact_type"
        )
    if (
        extraction.source_document_ref.artifact_id
        != source_document_artifact_id(
            extraction.project_id, extraction.document_id
        )
    ):
        raise StoryIntegrityError(
            "CandidateExtraction source_document_ref does not match its "
            "project/document identity"
        )
    if (
        extraction.source_chunk_ref.artifact_id
        != source_chunk_artifact_id(
            extraction.project_id,
            extraction.document_id,
            extraction.chunk_profile_id,
            extraction.chunk_id,
        )
    ):
        raise StoryIntegrityError(
            "CandidateExtraction source_chunk_ref does not match its "
            "project/document/chunk_profile/chunk identity"
        )
    return extraction


# ---------------------------------------------------------------------------
# Schema gate (single source of truth: candidate-extraction.schema.json)
# ---------------------------------------------------------------------------


def _validate_extraction_schema(serialized: dict[str, Any]) -> None:
    """Fail closed if the canonical serialized form fails the persisted schema."""
    schema = load_json(SCHEMAS_DIR / "candidate-extraction.schema.json")
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except Exception as exc:  # noqa: BLE001 - a broken tracked schema is a bug
        raise StoryIntegrityError(
            f"invalid candidate-extraction schema: {exc}"
        ) from exc
    errors = sorted(
        validator.iter_errors(serialized),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    if errors:
        error = errors[0]
        path = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise StoryIntegrityError(
            "CandidateExtraction canonical form fails "
            f"candidate-extraction.schema.json at {path}: {error.message}"
        )


# ---------------------------------------------------------------------------
# Semantic reuse identity (frozen contract section 21)
# ---------------------------------------------------------------------------


def _generation_semantic_identity(
    provenance: LLMInvocationProvenance,
) -> tuple[Any, ...]:
    """The A-I3 semantic identity fields (the authority for reuse identity).

    Provider metadata that is not part of the A3 semantic identity (endpoint,
    timeout, credential, provider response id, finish reason, usage) is
    deliberately excluded so runtime changes never invalidate reuse.
    """
    return (
        provenance.semantic_profile_id,
        provenance.semantic_profile_hash,
        provenance.prompt_id,
        provenance.prompt_version,
        provenance.prompt_content_hash,
        provenance.rendered_prompt_hash,
        provenance.output_schema_id,
        provenance.output_schema_version,
        provenance.output_schema_hash,
        provenance.request_hash,
    )


def extraction_semantic_identity(extraction: CandidateExtraction) -> tuple[Any, ...]:
    """The exact frozen semantic identity of a CandidateExtraction."""
    return (
        extraction.source_document_ref,
        extraction.source_chunk_ref,
        extraction.chunk_profile_id,
        extraction.extraction_profile_id,
        extraction.extraction_profile_hash,
        _generation_semantic_identity(extraction.generation_provenance),
    )


def requested_semantic_identity(
    *,
    source_document_ref: ArtifactRef,
    source_chunk_ref: ArtifactRef,
    chunk_profile_id: str,
    extraction_profile: StoryExtractionProfile,
    generation_provenance: LLMInvocationProvenance,
) -> tuple[Any, ...]:
    """The requested semantic identity, shaped identically to
    :func:`extraction_semantic_identity` for exact comparison."""
    return (
        source_document_ref,
        source_chunk_ref,
        chunk_profile_id,
        extraction_profile.profile_id,
        extraction_profile.profile_hash,
        _generation_semantic_identity(generation_provenance),
    )


# ---------------------------------------------------------------------------
# Source / chunk integrity
# ---------------------------------------------------------------------------


def _check_extraction_source_refs(
    *,
    source_document: SourceDocument,
    source_chunk: SourceChunk,
    source_document_ref: ArtifactRef,
    source_chunk_ref: ArtifactRef,
    chunk_profile_id: str,
) -> None:
    """Verify the pinned refs and the exact source pair agree on identity.

    ``chunk_profile_id`` must be consistent with the source-chunk artifact
    identity. A structurally inconsistent pair is an A1/A2 lineage/integrity
    failure and fails closed (not as an LLM regeneration finding).
    """
    project_id = source_chunk.project_id
    document_id = source_chunk.document_id
    chunk_id = source_chunk.chunk_id
    if (
        source_document.project_id != project_id
        or source_document.document_id != document_id
    ):
        raise StoryIntegrityError(
            "SourceDocument and SourceChunk project/document identity mismatch"
        )
    if source_document_ref.artifact_type != SOURCE_DOCUMENT_ARTIFACT_TYPE:
        raise StoryIntegrityError(
            "source_document_ref has the wrong artifact_type"
        )
    if source_chunk_ref.artifact_type != SOURCE_CHUNK_ARTIFACT_TYPE:
        raise StoryIntegrityError(
            "source_chunk_ref has the wrong artifact_type"
        )
    if (
        source_document_ref.artifact_id
        != source_document_artifact_id(project_id, document_id)
    ):
        raise StoryIntegrityError(
            "source_document_ref does not match the exact SourceDocument "
            "artifact identity"
        )
    if (
        source_chunk_ref.artifact_id
        != source_chunk_artifact_id(
            project_id, document_id, chunk_profile_id, chunk_id
        )
    ):
        raise StoryIntegrityError(
            "source_chunk_ref does not match the exact SourceChunk artifact "
            "identity for this chunk_profile_id/chunk_id"
        )


# ---------------------------------------------------------------------------
# Pointer / report helpers (mirror the A1/A2 hardening pattern)
# ---------------------------------------------------------------------------


def _current_pointer(
    pointer_store: FilePointerStore,
    pointer_id: str,
) -> tuple[ArtifactRef | None, ArtifactRef | None]:
    try:
        pointer_ref = pointer_store.resolve_current_pointer_ref(pointer_id)
        pointer = pointer_store.resolve_current(pointer_id)
        return pointer_ref, pointer.target_ref
    except PointerNotFoundError:
        return None, None


def _require_validation_report(
    store: FileArtifactStore,
    *,
    artifact_id: str,
    revision: int,
    expected_report: ValidationReport,
) -> ArtifactRef:
    """Verify the exact matching PASS A3 ValidationReport; fail closed on any
    missing / malformed / mismatched / non-PASS report (A1/A2 hardening)."""
    try:
        ref = store.get(
            VALIDATION_REPORT_ARTIFACT_TYPE, artifact_id, revision
        ).ref
        report = load_validation_report(store, ref)
        if report != expected_report:
            raise StoryIntegrityError(
                "A3 ValidationReport does not match the exact expected "
                "deterministic validation result"
            )
        if report.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "current CandidateExtraction is backed by a non-PASS "
                "A3 ValidationReport"
            )
        return ref
    except StoryIntegrityError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(
            "matching A3 ValidationReport is missing or invalid for "
            f"{artifact_id} revision {revision}"
        ) from exc


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateExtractionPublication:
    """Outcome of an A3C single-chunk persistence/reuse operation."""

    candidate_extraction_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    reused: bool


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CandidateExtractionService:
    """Narrow deterministic A3C service: persistence, reuse, and publication.

    A3D will call :meth:`extract_chunk` with an exact source pair, an
    extraction profile, the exact A-I3 generation provenance, and a candidate
    payload. The service never invokes an LLM and never regenerates; it only
    persists/reuses the validated immutable CandidateExtraction and publishes a
    matching PASS ``ValidationReport`` via compare-and-set.
    """

    def __init__(
        self,
        store: FileArtifactStore,
        pointers: FilePointerStore,
    ) -> None:
        if not isinstance(store, FileArtifactStore):
            raise StoryPersistenceError("store must be a FileArtifactStore")
        if not isinstance(pointers, FilePointerStore):
            raise StoryPersistenceError("pointers must be a FilePointerStore")
        self.store = store
        self.pointers = pointers

    def extract_chunk(
        self,
        *,
        source_document: SourceDocument,
        source_document_ref: ArtifactRef,
        source_chunk: SourceChunk,
        source_chunk_ref: ArtifactRef,
        chunk_profile_id: str,
        extraction_profile: StoryExtractionProfile,
        generation_provenance: LLMInvocationProvenance,
        payload: Any,
    ) -> CandidateExtractionPublication:
        """Extract (reuse or publish) the CandidateExtraction for one chunk.

        Reuse is current-only: only the exact CURRENT extraction is considered.
        A matching current extraction is re-verified (integrity, canonical
        payload, exact PASS ValidationReport, pointer re-check) and reused.
        Otherwise the requested payload is validated and, if PASS, a new
        revision is published under the same logical artifact identity.
        """
        project_id = source_chunk.project_id
        document_id = source_chunk.document_id
        chunk_id = source_chunk.chunk_id
        pointer_id = candidate_extraction_pointer_id(
            project_id,
            document_id,
            chunk_profile_id,
            chunk_id,
            extraction_profile.profile_id,
        )
        current_pointer_ref, current_extraction_ref = _current_pointer(
            self.pointers, pointer_id
        )

        # Current-only reuse: load the exact CURRENT target (fail closed) and
        # compare the frozen semantic identity.
        if current_extraction_ref is not None:
            current_extraction = load_candidate_extraction(
                self.store, current_extraction_ref
            )
            if extraction_semantic_identity(current_extraction) == (
                requested_semantic_identity(
                    source_document_ref=source_document_ref,
                    source_chunk_ref=source_chunk_ref,
                    chunk_profile_id=chunk_profile_id,
                    extraction_profile=extraction_profile,
                    generation_provenance=generation_provenance,
                )
            ):
                report_ref = self._verify_current_extraction(
                    current_extraction, current_extraction_ref
                )
                if (
                    self.pointers.resolve_current_pointer_ref(pointer_id)
                    != current_pointer_ref
                ):
                    raise StoryPersistenceError(
                        "CandidateExtraction CURRENT pointer changed during "
                        "reuse verification"
                    )
                assert current_pointer_ref is not None
                return CandidateExtractionPublication(
                    candidate_extraction_ref=current_extraction_ref,
                    validation_report_ref=report_ref,
                    current_pointer_ref=current_pointer_ref,
                    reused=True,
                )

        # No current, or a semantic-identity change: validate + publish a new
        # immutable revision (the old revision, if any, remains historical).
        return self._publish(
            project_id=project_id,
            document_id=document_id,
            chunk_profile_id=chunk_profile_id,
            chunk_id=chunk_id,
            source_document=source_document,
            source_document_ref=source_document_ref,
            source_chunk=source_chunk,
            source_chunk_ref=source_chunk_ref,
            extraction_profile=extraction_profile,
            generation_provenance=generation_provenance,
            payload=payload,
            pointer_id=pointer_id,
            current_pointer_ref=current_pointer_ref,
            current_extraction_ref=current_extraction_ref,
        )

    # -- reuse eligibility verification -----------------------------------

    def _verify_current_extraction(
        self,
        extraction: CandidateExtraction,
        extraction_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Re-verify the exact CURRENT extraction and its PASS ValidationReport.

        Loads the pinned source pair (authoritative), re-runs the deterministic
        A3B semantic validation, verifies the canonical payload, and compares
        the exact matching ValidationReport. Any corruption of an object
        currently claimed by CURRENT fails closed.
        """
        source_document = load_source_document(
            self.store, extraction.source_document_ref
        )
        source_chunk = load_source_chunk(
            self.store, extraction.source_chunk_ref
        )
        _check_extraction_source_refs(
            source_document=source_document,
            source_chunk=source_chunk,
            source_document_ref=extraction.source_document_ref,
            source_chunk_ref=extraction.source_chunk_ref,
            chunk_profile_id=extraction.chunk_profile_id,
        )

        result = validate_candidate_payload(
            extraction.candidates, source_document, source_chunk
        )
        if result.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "persisted CandidateExtraction does not re-validate to PASS; "
                "it is not current-eligible"
            )
        if (
            canonicalize_candidate_payload(
                extraction.candidates, source_document
            )
            != extraction.candidates
        ):
            raise StoryIntegrityError(
                "persisted CandidateExtraction payload is not in "
                "deterministic canonical form"
            )

        expected_report = ValidationReport(
            validated_refs=(
                LineageRef("source_document", extraction.source_document_ref),
                LineageRef("source_chunk", extraction.source_chunk_ref),
                LineageRef("candidate_extraction", extraction_ref),
            ),
            findings=result.findings,
        )
        return _require_validation_report(
            self.store,
            artifact_id=candidate_extraction_validation_artifact_id(
                extraction_ref.artifact_id
            ),
            revision=extraction_ref.revision,
            expected_report=expected_report,
        )

    # -- publication -------------------------------------------------------

    def _publish(
        self,
        *,
        project_id: str,
        document_id: str,
        chunk_profile_id: str,
        chunk_id: str,
        source_document: SourceDocument,
        source_document_ref: ArtifactRef,
        source_chunk: SourceChunk,
        source_chunk_ref: ArtifactRef,
        extraction_profile: StoryExtractionProfile,
        generation_provenance: LLMInvocationProvenance,
        payload: Any,
        pointer_id: str,
        current_pointer_ref: ArtifactRef | None,
        current_extraction_ref: ArtifactRef | None,
    ) -> CandidateExtractionPublication:
        # The pinned refs must point at the exact in-memory source pair.
        _check_extraction_source_refs(
            source_document=source_document,
            source_chunk=source_chunk,
            source_document_ref=source_document_ref,
            source_chunk_ref=source_chunk_ref,
            chunk_profile_id=chunk_profile_id,
        )

        # Deterministic A3B semantic validation (fails closed on a
        # structurally inconsistent source pair; produces findings otherwise).
        result = validate_candidate_payload(
            payload, source_document, source_chunk
        )
        if result.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "A3 semantic validation failed; CandidateExtraction cannot "
                "become current"
            )
        canonical_payload = result.canonical_payload
        assert canonical_payload is not None

        extraction = CandidateExtraction(
            schema_version=CANDIDATE_EXTRACTION_SCHEMA_VERSION,
            project_id=project_id,
            document_id=document_id,
            chunk_profile_id=chunk_profile_id,
            chunk_id=chunk_id,
            source_document_ref=source_document_ref,
            source_chunk_ref=source_chunk_ref,
            extraction_profile_id=extraction_profile.profile_id,
            extraction_profile_hash=extraction_profile.profile_hash,
            generation_provenance=generation_provenance,
            candidates=canonical_payload,
        )
        # Gate the canonical serialized form against the persisted schema before
        # it can become current.
        _validate_extraction_schema(extraction.to_dict())

        revision = next_candidate_extraction_revision(
            self.store,
            project_id=project_id,
            document_id=document_id,
            chunk_profile_id=chunk_profile_id,
            chunk_id=chunk_id,
            extraction_profile_id=extraction_profile.profile_id,
            current_target_ref=current_extraction_ref,
        )
        extraction_ref = persist_candidate_extraction(
            self.store, extraction, revision=revision
        )

        report = ValidationReport(
            validated_refs=(
                LineageRef("source_document", source_document_ref),
                LineageRef("source_chunk", source_chunk_ref),
                LineageRef("candidate_extraction", extraction_ref),
            ),
            findings=result.findings,
        )
        if report.summary.result is not ValidationResult.PASS:
            raise StoryIntegrityError(
                "A3 ValidationReport is not PASS; CandidateExtraction cannot "
                "become current"
            )
        report_ref = persist_validation_report(
            self.store,
            report,
            artifact_id=candidate_extraction_validation_artifact_id(
                extraction_ref.artifact_id
            ),
            revision=revision,
        )
        # Verify the persisted report round-trips to the exact PASS result.
        if (
            load_validation_report(self.store, report_ref).summary.result
            is not ValidationResult.PASS
        ):
            raise StoryIntegrityError(
                "persisted A3 ValidationReport is not PASS"
            )

        try:
            pointer_ref = self.pointers.compare_and_set(
                pointer_id=pointer_id,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=current_pointer_ref,
                target_ref=extraction_ref,
            )
        except Exception as exc:  # noqa: BLE001
            raise StoryPersistenceError(
                "failed to publish CandidateExtraction CURRENT pointer; "
                "immutable artifacts remain historical"
            ) from exc

        return CandidateExtractionPublication(
            candidate_extraction_ref=extraction_ref,
            validation_report_ref=report_ref,
            current_pointer_ref=pointer_ref,
            reused=False,
        )
