from __future__ import annotations


class StoryError(Exception):
    """Base error for deterministic v1.2 Story Analysis stages."""


class StoryConfigError(StoryError):
    """Project/profile configuration is invalid."""


class SourceIngestionError(StoryError):
    """A1 source ingestion failed."""


class SourceReadError(SourceIngestionError):
    """Source bytes cannot be read."""


class SourceDecodeError(SourceIngestionError):
    """Source text cannot be decoded under the selected policy."""


class SourceStructureError(SourceIngestionError):
    """Parsed source structure violates the A1 contract."""


class SourcePdfError(SourceIngestionError):
    """Text-based PDF extraction failed or yielded no usable text."""


class SourceLanguageError(SourceIngestionError):
    """Source-language detection failed."""


class StoryPersistenceError(StoryError):
    """Typed Story artifact persistence or resolution failed."""


class StoryIntegrityError(StoryPersistenceError):
    """Persisted Story artifact violates its typed contract."""


class ChunkPlanningError(StoryError):
    """A2 chunk planning failed."""


class ChunkProfileError(ChunkPlanningError):
    """Chunk-planning profile is invalid."""


class ChunkCoverageError(ChunkPlanningError):
    """Chunk ownership/context coverage violates A2 invariants."""


class ExtractionModelError(StoryError):
    """A3 candidate-extraction domain/profile contract is structurally invalid.

    This is the static, object-level domain error for A3A. It is deliberately
    distinct from A-I3 LLM errors (transport/config/provenance) and from the
    A3B semantic source/ownership/cross-reference validation findings, which
    belong to a later slice.
    """


class ReconciliationModelError(StoryError):
    """A4 entity-reconciliation domain/profile contract is structurally invalid.

    This is the static, object-level domain error for A4A. It is deliberately
    distinct from the A3A extraction model error, from A-I3 LLM errors
    (transport/config/provenance), and from the later A4B-A4E reconciliation
    validation findings (coverage / graph / persistence), which belong to later
    slices.
    """


class ExtractionProvenanceError(StoryError):
    """A successful structured-generation result's provenance does not
    correspond to the exact semantic request that produced it.

    This is an A3D integrity failure (not an A-I3 transport error and not an A3
    semantic-validation finding): a fake/broken client that returns provenance
    from a different request must fail closed and never be published. A3D does
    NOT route this into a semantic-regeneration round.
    """


class ExtractionSemanticGenerationError(StoryError):
    """Both permitted A3 semantic generation rounds produced JSON-Schema-valid
    but semantically-invalid payloads; no CandidateExtraction could be
    published.

    Carries deterministic diagnostics for tests/review. CURRENT is unchanged and
    no failed payload is persisted. This is a *semantic* failure, not a
    transport error: the underlying A-I3 provider calls all succeeded.
    """

    def __init__(
        self,
        *,
        rounds_attempted: int,
        final_findings: tuple,
        final_validation_result,
    ) -> None:
        self.rounds_attempted = rounds_attempted
        self.final_findings = tuple(final_findings)
        self.final_validation_result = final_validation_result
        super().__init__(
            "A3 semantic validation failed after "
            f"{rounds_attempted} semantic generation round(s); no "
            "CandidateExtraction could be published"
        )
