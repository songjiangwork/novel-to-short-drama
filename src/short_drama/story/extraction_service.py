"""v1.2 A3D — single-chunk semantic extraction orchestration.

This module is the *orchestration* layer for A3 (chunk extraction). It sits
strictly on top of the merged A3A domain models, A3B semantic validation, and
A3C persistence/reuse, and drives the provider-neutral A-I3
:class:`LLMClient`. For one exact chunk it answers:

  * resolve the exact A1 ``SourceDocument`` + A2 ``SourceChunk`` and verify
    their lineage/identity BEFORE any provider call (a structural or lineage
    contradiction fails closed as a Story integrity error and never consumes an
    LLM attempt);
  * deterministically partition the chunk into LEFT / OWNERSHIP / RIGHT context
    and render the versioned ``a3.chunk-extraction`` prompt;
  * build the A-I3 ``OutputSchema`` from the tracked
    ``candidate-payload.schema.json`` with the ``StoryExtractionProfile``-pinned
    identity, and enforce a mandatory cross-contract consistency gate
    (profile <-> prompt, profile <-> output schema) before any reuse or LLM
    call;
  * build the A-I3 ``StructuredGenerationRequest`` (the single semantic
    request identity reused for the pre-generation reuse check and every
    semantic generation round);
  * short-circuit via the merged A3C pre-generation reuse check *before* any
    provider invocation (a current-eligible exact-identity match is returned
    with zero provider calls);
  * on a miss, call :meth:`LLMClient.generate_structured` (which owns the A-I3
    technical retry budget of up to 3 provider attempts), typed-load the result
    through ``CandidatePayload.from_dict``, run A3B semantic validation, and
    publish through A3C ``publish_validated``;
  * bound semantic regeneration to ``StoryExtractionProfile.max_generation_rounds``
    (frozen v1 ceiling of 2): a JSON-Schema-valid but semantically-invalid
    payload consumes one round and, if one remains, is regenerated with the
    *exact same* semantic request (no adaptive repair prompt, no changed
    temperature/schema/system message).

A-I3 vs A3D retry discipline (frozen contract section 18): one call to
``generate_structured`` may internally consume up to 3 A-I3 technical attempts;
A3D treats that as a single semantic generation round and does NOT wrap
``LLMError`` in a second retry loop. If A-I3 exhausts its own budget and raises
``LLMRetryExhaustedError`` (or any other non-retryable ``LLMError``), it is
propagated unchanged — A3D does not start another semantic round for it. Only a
*successful* ``generate_structured`` return that is locally JSON-Schema-valid
may trigger an A3D semantic round. The theoretical provider ceiling stays
``2 semantic rounds x 3 A-I3 attempts = 6``.

Semantic uncertainty (``unresolved_mentions``, ``evidence_strength=uncertain``,
``temporal_mode=unknown``, empty ``possible_candidate_refs``) is a *successful*
domain result: when A3B says PASS the extraction is published and is never
regenerated for being uncertain.

Deliberately out of scope (later slices): multi-chunk iteration, the
``extract-chunks`` batch CLI, batch summary, A4 reconciliation, A5
consolidation, A6 StoryBible, provider routing/fallback, any generic LLM
cache, and any raw-LLM response persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.artifacts import (
    ArtifactRef,
    FileArtifactStore,
    canonical_json_bytes,
)
from short_drama.io import load_json
from short_drama.llm import (
    LLMClient,
    OutputSchema,
    PromptRegistry,
    PromptSpec,
    SemanticLLMProfile,
    build_structured_request,
    render_prompt,
)
from short_drama.paths import REPO_ROOT, SCHEMAS_DIR

from .chunking import SourceChunk
from .errors import (
    ExtractionModelError,
    ExtractionProvenanceError,
    ExtractionSemanticGenerationError,
    StoryConfigError,
    StoryIntegrityError,
)
from .extraction import CandidatePayload, StoryExtractionProfile
from .extraction_persistence import (
    CandidateExtractionPublication,
    CandidateExtractionService,
    request_semantic_fields,
)
from .extraction_validation import validate_candidate_payload
from .persistence import (
    load_source_chunk,
    load_source_document,
    source_chunk_artifact_id,
    source_document_artifact_id,
)
from .source import SourceDocument

# Deterministic default authorities (tracked prompt registry + tracked output
# schema). Both are overridable for tests; production uses the tracked files.
DEFAULT_PROMPT_BASE_DIR = REPO_ROOT / "prompts" / "story"
DEFAULT_OUTPUT_SCHEMA_PATH = SCHEMAS_DIR / "candidate-payload.schema.json"


# ---------------------------------------------------------------------------
# Deterministic LEFT / OWNERSHIP / RIGHT context partition
# ---------------------------------------------------------------------------


def partition_paragraph_ids(
    source_chunk: SourceChunk,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split ``SourceChunk.paragraph_ids`` into
    (LEFT, OWNERSHIP, RIGHT) paragraph-id lists.

    ``SourceChunk.paragraph_ids`` is the ordered context authority and
    ``ownership_span.start..end`` is inclusive. Everything before the ownership
    start is LEFT, the ownership span itself is OWNERSHIP (always non-empty),
    and everything after the ownership end is RIGHT. A structurally
    inconsistent ownership span fails closed as a Story integrity error (never
    as an LLM regeneration finding).
    """
    if not isinstance(source_chunk, SourceChunk):
        raise StoryIntegrityError("source_chunk must be a SourceChunk")
    paragraph_ids = list(source_chunk.paragraph_ids)
    start = source_chunk.ownership_span.start
    end = source_chunk.ownership_span.end
    if start not in paragraph_ids:
        raise StoryIntegrityError(
            f"ownership_span.start {start!r} is not in SourceChunk.paragraph_ids"
        )
    if end not in paragraph_ids:
        raise StoryIntegrityError(
            f"ownership_span.end {end!r} is not in SourceChunk.paragraph_ids"
        )
    start_index = paragraph_ids.index(start)
    end_index = paragraph_ids.index(end)
    if start_index > end_index:
        raise StoryIntegrityError("SourceChunk.ownership_span is reversed")
    left = paragraph_ids[:start_index]
    ownership = paragraph_ids[start_index : end_index + 1]
    right = paragraph_ids[end_index + 1 :]
    if not ownership:
        raise StoryIntegrityError("OWNERSHIP span must be non-empty")
    return left, ownership, right


def paragraphs_to_canonical_json(
    source_document: SourceDocument, paragraph_ids: list[str]
) -> str:
    """Render an ordered list of paragraph ids as a deterministic canonical
    JSON array string:

    ``[{"paragraph_id": "...", "text_original": "..."}, ...]``

    Uses the shared RFC 8785 canonical serialization authority (stable key
    order, no incidental ``json.dumps`` formatting) and the exact
    ``SourceDocument`` ``text_original``. The array order is preserved verbatim
    (source order). Empty ``paragraph_ids`` render as ``[]``.
    """
    if not isinstance(source_document, SourceDocument):
        raise StoryIntegrityError(
            "source_document must be an exact A1 SourceDocument"
        )
    paragraph_index = source_document.paragraph_index()
    items: list[dict[str, str]] = []
    for paragraph_id in paragraph_ids:
        paragraph = paragraph_index.get(paragraph_id)
        if paragraph is None:
            raise StoryIntegrityError(
                f"paragraph {paragraph_id!r} is not present in the exact "
                "SourceDocument"
            )
        items.append(
            {"paragraph_id": paragraph_id, "text_original": paragraph.text_original}
        )
    return canonical_json_bytes(items).decode("utf-8")


@dataclass(frozen=True, slots=True)
class ChunkContext:
    """The exact prompt render variables for a single chunk."""

    chunk_id: str
    left_context_json: str
    ownership_json: str
    right_context_json: str

    def as_variables(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "left_context_json": self.left_context_json,
            "ownership_json": self.ownership_json,
            "right_context_json": self.right_context_json,
        }


def build_chunk_context(
    source_document: SourceDocument, source_chunk: SourceChunk
) -> ChunkContext:
    """Build the deterministic LEFT / OWNERSHIP / RIGHT render variables for the
    exact source pair. LEFT/RIGHT may be empty (``[]``); OWNERSHIP is non-empty.

    No prior ``CandidateExtraction`` or other chunk's semantic output is ever
    injected: the prompt carries only this chunk's exact source context.
    """
    left_ids, ownership_ids, right_ids = partition_paragraph_ids(source_chunk)
    return ChunkContext(
        chunk_id=source_chunk.chunk_id,
        left_context_json=paragraphs_to_canonical_json(source_document, left_ids),
        ownership_json=paragraphs_to_canonical_json(source_document, ownership_ids),
        right_context_json=paragraphs_to_canonical_json(source_document, right_ids),
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ChunkExtractionService:
    """Narrow deterministic A3D single-chunk semantic extraction service.

    Holds the exact source/persistence authorities (``FileArtifactStore`` +
    ``FilePointerStore``), the versioned prompt registry, and the tracked output
    schema path, and owns the merged A3C ``CandidateExtractionService`` as its
    persistence/reuse authority. It knows nothing about base_url, HTTP,
    credentials, a concrete model, or a provider-specific request field: it
    depends only on the provider-neutral ``LLMClient``.

    ``store``/``pointers`` are passed so A3D can exact-resolve the requested
    source pair and so A3C can persist/reuse under the same physical
    authorities (single store, single pointer head).
    """

    def __init__(
        self,
        store: FileArtifactStore,
        pointers,
        prompt_registry: PromptRegistry | None = None,
        output_schema_path: str | Path = DEFAULT_OUTPUT_SCHEMA_PATH,
    ) -> None:
        self.store = store
        self.pointers = pointers
        self.prompt_registry = (
            prompt_registry if prompt_registry is not None else PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
        )
        self.output_schema_path = Path(output_schema_path)
        self._persistence = CandidateExtractionService(store, pointers)

    # -- public entry point --------------------------------------------------

    def extract_chunk(
        self,
        *,
        source_document_ref: ArtifactRef,
        source_chunk_ref: ArtifactRef,
        chunk_profile_id: str,
        extraction_profile: StoryExtractionProfile,
        semantic_profile: SemanticLLMProfile,
        llm_client: LLMClient,
    ) -> CandidateExtractionPublication:
        """Extract one exact chunk and publish its validated CandidateExtraction.

        Returns a :class:`CandidateExtractionPublication`. On a pre-generation
        reuse hit it returns the existing publication with zero provider calls;
        on a miss it runs at most ``extraction_profile.max_generation_rounds``
        semantic generation rounds and publishes through A3C. Fails closed
        (raising) on any source/lineage, profile/prompt/schema, or provenance
        integrity contradiction; raises
        :class:`ExtractionSemanticGenerationError` if every permitted semantic
        round is semantically invalid; propagates any A-I3 ``LLMError``
        unchanged.
        """
        # 1. Exact source resolution + lineage (fail closed, no LLM call).
        source_document, source_chunk = self._resolve_exact_source_pair(
            source_document_ref=source_document_ref,
            source_chunk_ref=source_chunk_ref,
            chunk_profile_id=chunk_profile_id,
        )

        # 2. Load the profile-pinned prompt and build the tracked output schema.
        prompt_spec = self._load_prompt_spec(extraction_profile)
        output_schema = self._build_output_schema(extraction_profile)

        # 3. Mandatory cross-contract consistency gate (fail closed, no LLM call).
        self._check_profile_prompt_schema_consistency(
            extraction_profile, prompt_spec, output_schema
        )

        # 4. Deterministic context partition + prompt rendering.
        context = build_chunk_context(source_document, source_chunk)
        rendered_prompt = render_prompt(prompt_spec, context.as_variables())

        # 5. Build the single semantic request identity (reuse + every round).
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )

        # 6. Pre-generation reuse (A3C) BEFORE any provider invocation.
        reused = self._persistence.try_reuse_current(
            project_id=source_chunk.project_id,
            document_id=source_chunk.document_id,
            chunk_profile_id=chunk_profile_id,
            chunk_id=source_chunk.chunk_id,
            source_document_ref=source_document_ref,
            source_chunk_ref=source_chunk_ref,
            extraction_profile=extraction_profile,
            structured_request=request,
        )
        if reused is not None:
            return reused

        # 7. Bounded semantic generation rounds (A3D owns this bound only).
        max_rounds = extraction_profile.max_generation_rounds
        last_findings: tuple = ()
        last_validation_result = None
        for _round_number in range(1, max_rounds + 1):
            # 7a. One provider-neutral generation (A-I3 owns the technical retry).
            result = llm_client.generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )
            # 7b. Provenance/request consistency (fail closed, never publish).
            self._verify_provenance_matches_request(result.provenance, request)
            # 7c. Typed domain load: a JSON-Schema-valid object the A3 typed
            #     model still rejects is a semantic-invalid round (it consumes
            #     this round; nothing is persisted for it).
            try:
                payload = CandidatePayload.from_dict(result.parsed_json)
            except ExtractionModelError:
                last_findings = ()
                last_validation_result = None
                continue
            # 7d. A3B semantic validation.
            validation = validate_candidate_payload(
                payload, source_document, source_chunk
            )
            last_findings = validation.findings
            last_validation_result = validation.summary
            if not validation.is_valid:
                continue
            # 7e. A3C publish (persistence authority): A3C validates, canonicalizes,
            #     persists the immutable revision, publishes the exact PASS
            #     ValidationReport, and compare-and-set moves CURRENT. A
            #     publish-time same-identity race is honored (``reused=True``).
            return self._persistence.publish_validated(
                source_document=source_document,
                source_document_ref=source_document_ref,
                source_chunk=source_chunk,
                source_chunk_ref=source_chunk_ref,
                chunk_profile_id=chunk_profile_id,
                extraction_profile=extraction_profile,
                generation_provenance=result.provenance,
                payload=payload,
            )

        # 8. Every permitted round was semantically invalid.
        raise ExtractionSemanticGenerationError(
            rounds_attempted=max_rounds,
            final_findings=last_findings,
            final_validation_result=last_validation_result,
        )

    # -- private steps -------------------------------------------------------

    def _resolve_exact_source_pair(
        self,
        *,
        source_document_ref: ArtifactRef,
        source_chunk_ref: ArtifactRef,
        chunk_profile_id: str,
    ) -> tuple[SourceDocument, SourceChunk]:
        """Exact-resolve the requested source pair through the authoritative
        A1/A2 typed loaders and verify its lineage/identity.

        Any structural or lineage contradiction is an A1/A2 integrity failure:
        it fails closed before any LLM call and is never disguised as a
        semantic-regeneration attempt.
        """
        source_document = load_source_document(self.store, source_document_ref)
        source_chunk = load_source_chunk(self.store, source_chunk_ref)
        project_id = source_document.project_id
        document_id = source_document.document_id
        if (
            source_chunk.project_id != project_id
            or source_chunk.document_id != document_id
        ):
            raise StoryIntegrityError(
                "SourceChunk project/document identity does not match the exact "
                "SourceDocument"
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
                project_id, document_id, chunk_profile_id, source_chunk.chunk_id
            )
        ):
            raise StoryIntegrityError(
                "source_chunk_ref does not match the exact SourceChunk artifact "
                "identity for this chunk_profile_id/chunk_id"
            )
        if source_chunk.source_document_ref != source_document_ref:
            raise StoryIntegrityError(
                "SourceChunk.source_document_ref does not exactly match "
                "source_document_ref"
            )
        paragraph_index = source_document.paragraph_index()
        for paragraph_id in source_chunk.paragraph_ids:
            if paragraph_id not in paragraph_index:
                raise StoryIntegrityError(
                    f"SourceChunk paragraph {paragraph_id!r} is not present in "
                    "the exact SourceDocument"
                )
        return source_document, source_chunk

    def _load_prompt_spec(
        self, extraction_profile: StoryExtractionProfile
    ) -> PromptSpec:
        """Load exactly the prompt pinned by the StoryExtractionProfile."""
        return self.prompt_registry.load(
            extraction_profile.prompt_id,
            version=extraction_profile.prompt_version,
        )

    def _build_output_schema(
        self, extraction_profile: StoryExtractionProfile
    ) -> OutputSchema:
        """Build the A-I3 ``OutputSchema`` from the tracked candidate-payload
        schema, with the ``StoryExtractionProfile``-pinned identity."""
        try:
            schema = load_json(self.output_schema_path)
        except Exception as exc:  # noqa: BLE001 - a broken tracked schema is a config error
            raise StoryConfigError(
                f"failed to load candidate-payload output schema: {exc}"
            ) from exc
        if not isinstance(schema, dict):
            raise StoryConfigError(
                "candidate-payload output schema must be a JSON object"
            )
        return OutputSchema.create(
            schema_id=extraction_profile.output_schema_id,
            schema_version=extraction_profile.output_schema_version,
            schema=schema,
        )

    def _check_profile_prompt_schema_consistency(
        self,
        extraction_profile: StoryExtractionProfile,
        prompt_spec: PromptSpec,
        output_schema: OutputSchema,
    ) -> None:
        """Fail closed unless the profile, the loaded prompt, and the built
        output schema all agree. A contradiction is a deterministic
        config/prompt failure (no LLM retry)."""
        if extraction_profile.prompt_id != prompt_spec.prompt_id:
            raise StoryConfigError(
                "StoryExtractionProfile.prompt_id "
                f"({extraction_profile.prompt_id!r}) does not match the loaded "
                f"PromptSpec.prompt_id ({prompt_spec.prompt_id!r})"
            )
        if extraction_profile.prompt_version != prompt_spec.version:
            raise StoryConfigError(
                "StoryExtractionProfile.prompt_version "
                f"({extraction_profile.prompt_version!r}) does not match the "
                f"loaded PromptSpec.version ({prompt_spec.version!r})"
            )
        if extraction_profile.output_schema_id != output_schema.schema_id:
            raise StoryConfigError(
                "StoryExtractionProfile.output_schema_id "
                f"({extraction_profile.output_schema_id!r}) does not match the "
                f"OutputSchema.schema_id ({output_schema.schema_id!r})"
            )
        if extraction_profile.output_schema_version != output_schema.schema_version:
            raise StoryConfigError(
                "StoryExtractionProfile.output_schema_version "
                f"({extraction_profile.output_schema_version!r}) does not match "
                f"the OutputSchema.schema_version "
                f"({output_schema.schema_version!r})"
            )

    def _verify_provenance_matches_request(self, provenance, request) -> None:
        """Verify the structured-generation provenance corresponds to the exact
        request that was built, using the shared A3C semantic-field authority.
        A mismatch fails closed (never published, never regenerated)."""
        if request_semantic_fields(provenance) != request_semantic_fields(request):
            raise ExtractionProvenanceError(
                "structured-generation provenance does not match the built "
                "semantic request (request/provenance semantic identity "
                "mismatch); result is not published"
            )


__all__ = [
    "ChunkContext",
    "ChunkExtractionService",
    "DEFAULT_OUTPUT_SCHEMA_PATH",
    "DEFAULT_PROMPT_BASE_DIR",
    "build_chunk_context",
    "paragraphs_to_canonical_json",
    "partition_paragraph_ids",
]
