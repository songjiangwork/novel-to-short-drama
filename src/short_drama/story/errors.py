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


class ConsolidationModelError(StoryError):
    """A5 fact/event/relationship consolidation domain/profile contract is
    structurally invalid.

    This is the static, object-level domain error for A5A. It is deliberately
    distinct from the A4A reconciliation model error, from A-I3 LLM errors
    (transport/config/provenance), and from the later A5B-A5H consolidation
    validation findings (blocking / semantic / canonical / coverage / reuse),
    which belong to later slices.
    """


class StoryAnalysisModelError(StoryError):
    """A6A static Story Analysis domain/profile contract is invalid."""


class StoryAnalysisPlanningError(StoryError):
    """A6B deterministic Story Analysis planning invariant is violated.

    This is the object-level planning error for A6B: a budget that the frozen
    policy cannot satisfy (which must FAIL CLOSED rather than truncate), a
    window ownership/coverage breach, or an unknown canonical ref encountered
    while building a deterministic plan. It is deliberately distinct from the
    A6A static-contract model error and from the A5 persistence/integrity
    errors (which surface while resolving the exact A5 CURRENT snapshot).
    """


class StoryAnalysisSemanticError(StoryError):
    """Base error for A6C character analysis semantic-pass failures.

    A6C executes the frozen A6C ``character_analysis`` semantic pass (prompt
    ``a6.character-analysis`` v1, output schema
    ``a6-character-analysis-output``, semantic profile ``story-analysis-llm-v1``)
    over the complete ordered A6B planned character evidence universe, and
    produces a complete in-memory :class:`~short_drama.story.CharacterAnalysisSet`.

    These are the A6 semantic-layer failure boundaries: provenance mismatch
    (fail closed, no semantic retry), semantic regeneration exhaustion (the
    two bounded A6C rounds both produced a schema-valid but semantically
    invalid result), and complete-character-coverage mismatch. Technical LLM
    failures (transport/HTTP/timeout, malformed JSON, and schema-invalid
    output) remain owned by A-I3 (``short_drama.llm``) and are never
    translated into A6C semantic errors.
    """


class StoryAnalysisProvenanceError(StoryAnalysisSemanticError):
    """A6C provenance verification failed; fail closed, no semantic retry.

    The provider boundary returned a result whose backend-neutral
    semantic/request identity does not match the exact character request
    identity (semantic profile, prompt, rendered prompt, output schema, or
    request hash). A6C must never accept such a result and must never route it
    into a semantic regeneration round.
    """


class StoryAnalysisSemanticGenerationError(StoryAnalysisSemanticError):
    """A6C semantic regeneration exhausted after max_generation_rounds (2).

    All permitted A6C semantic generation rounds for a single character
    produced schema-valid provider results that were rejected by typed load /
    exact evidence-ref validation; a CharacterAnalysis for that character could
    not be produced. A6C is in-memory only, so no partial character analysis is
    published. This is a *semantic* failure, not a transport error: the
    underlying A-I3 provider calls all succeeded.

    Carries deterministic diagnostics for tests/review: the affected
    ``character_ref``, the backend-neutral ``request_hash``, the number of
    semantic rounds attempted, and the bounded failure details from the final
    failed round.
    """

    def __init__(
        self,
        *,
        character_ref: str,
        request_hash: str,
        rounds_attempted: int,
        last_failure_details: tuple[str, ...],
    ) -> None:
        self.character_ref = character_ref
        self.request_hash = request_hash
        self.rounds_attempted = rounds_attempted
        self.last_failure_details = last_failure_details
        super().__init__(
            f"A6C character analysis for {character_ref!r} failed after "
            f"{rounds_attempted} semantic generation round(s): "
            f"{'; '.join(last_failure_details)}"
        )


class ConsolidationSemanticError(StoryError):
    """A5C fact semantic ambiguity resolution failure.

    Base error for the A5C-B provider-neutral fact semantic resolution slice.
    It is deliberately distinct from A5A model errors, A5B planning errors,
    and A-I3 LLM transport errors.
    """


class ConsolidationProvenanceError(ConsolidationSemanticError):
    """A successful structured-generation result's provenance does not
    correspond to the exact semantic request that produced it.

    This is an A5C integrity failure (not an A-I3 transport error): a
    fake/broken client that returns provenance from a different request must
    fail closed and never be published. A5C does NOT route this into a
    semantic-regeneration round (FAIL CLOSED, no retry).
    """


class ConsolidationSemanticGenerationError(ConsolidationSemanticError):
    """All permitted A5C fact semantic generation rounds produced valid
    provider results that were rejected by typed payload / pair / selector
    validation; no block decisions could be produced.

    Carries deterministic diagnostics for tests/review. A5C does not persist,
    so no partially-current A5 state exists. This is a *semantic* failure, not
    a transport error: the underlying A-I3 provider calls all succeeded.
    """

    def __init__(
        self,
        *,
        block_id: str,
        request_hash: str,
        rounds_attempted: int,
        last_failure_details: str,
        expected_pairs: tuple[tuple[str, str], ...],
    ) -> None:
        self.block_id = block_id
        self.request_hash = request_hash
        self.rounds_attempted = rounds_attempted
        self.last_failure_details = last_failure_details
        self.expected_pairs = expected_pairs
        super().__init__(
            f"A5C fact semantic block {block_id!r} failed after "
            f"{rounds_attempted} semantic generation round(s): "
            f"{last_failure_details}"
        )


class ConsolidationCurrentMissingError(StoryError):
    """A5 requires a current-eligible A4 CURRENT that does not exist.

    A5B (and the A5 stage in general) consumes the A4 EntityMap strictly
    through its CURRENT pointer. A missing A4 CURRENT is a structural A5
    failure, not a normal cache miss: A4 is a hard dependency of A5, so the
    absent CURRENT must fail closed rather than be treated as "not ready yet".
    This is deliberately distinct from the A4 ``try_reuse_current`` cache-miss
    (which returns ``None``) and from A5A/A5B domain-model errors.
    """


class ConsolidationUpstreamUnstableError(StoryError):
    """The upstream A4 CURRENT advanced during an A5 run; publication fails closed.

    A5F2 publishes an A5 state against the exact A4 EntityMap + A3 input identity
    that the A5 run consumed. If the A4 CURRENT pointer (or its A3 input identity)
    advances between the A5 run start and the publication CAS, the in-memory A5
    state is stale and must never be published. This is an A5F2 upstream-stability
    failure, distinct from the A5B missing-CURRENT failure
    (``ConsolidationCurrentMissingError``) and from A5A domain-model errors.
    """


class ReconciliationPlanningError(StoryError):
    """A4B deterministic reconciliation planning failed.

    This is the object-level error for A4B: snapshot coherence violation,
    coverage audit failure, or pair-planning structural invariant breach.
    It is deliberately distinct from A4A model errors and from the later
    A4C-A4E semantic/graph/persistence findings.
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


class ReconciliationSemanticError(StoryError):
    """A4C semantic ambiguity resolution failure.

    Base error for the A4C provider-neutral semantic resolution slice. It is
    deliberately distinct from A4A model errors, A4B planning errors, and
    A-I3 LLM transport errors.
    """


class ReconciliationProvenanceError(ReconciliationSemanticError):
    """A successful structured-generation result's provenance does not
    correspond to the exact semantic request that produced it.

    This is an A4C integrity failure (not an A-I3 transport error): a
    fake/broken client that returns provenance from a different request must
    fail closed and never be published. A4C does NOT route this into a
    semantic-regeneration round.
    """


class ReconciliationSemanticGenerationError(ReconciliationSemanticError):
    """All permitted A4C semantic generation rounds produced valid provider
    results that were rejected by exact pair/evidence validation; no block
    decisions could be produced.

    Carries deterministic diagnostics for tests/review. A4C does not persist,
    so no partially-current A4 state exists. This is a *semantic* failure, not
    a transport error: the underlying A-I3 provider calls all succeeded.
    """

    def __init__(
        self,
        *,
        block_id: str,
        rounds_attempted: int,
        last_failure_details: str,
        expected_pairs: tuple[tuple[str, str], ...],
    ) -> None:
        self.block_id = block_id
        self.rounds_attempted = rounds_attempted
        self.last_failure_details = last_failure_details
        self.expected_pairs = expected_pairs
        super().__init__(
            f"A4C semantic block {block_id!r} failed after "
            f"{rounds_attempted} semantic generation round(s): "
            f"{last_failure_details}"
        )
