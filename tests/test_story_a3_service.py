"""A3D single-chunk semantic extraction orchestration tests.

Covers the frozen A3D contract (parent A-I4 sections 15-24, 28) against a
deterministic fake ``LLMClient`` — NO live Qwen server, NO real provider.

  * deterministic LEFT / OWNERSHIP / RIGHT partition + canonical paragraph
    JSON (empty LEFT/RIGHT accepted, non-empty OWNERSHIP, source text does not
    become template syntax);
  * exact source resolution / lineage failures fail closed BEFORE any LLM call;
  * mandatory cross-contract consistency gate (profile <-> prompt, profile <->
    output schema) fails closed before any LLM call;
  * stable request hash;
  * pre-generation A3C reuse (hit -> zero provider calls; stale identity ->
    provider called; corrupt CURRENT -> fail closed, zero provider calls);
  * first valid structured result -> one semantic round + A3C publication;
  * bounded semantic regeneration (round 1 invalid -> round 2 valid -> success;
    both invalid -> bounded failure; previous valid CURRENT preserved; both
    rounds use the exact same request hash; no adaptive repair prompt);
  * semantic uncertainty is SUCCESS (no retry, one provider call);
  * A-I3 retry delegation (LLMRetryExhaustedError / non-retryable LLMError are
    propagated unchanged, no second semantic round, no nested retry);
  * provenance/request consistency verification (match succeeds, mismatch fails
    closed and is never persisted);
  * same-identity publish race (A3C ``reused=True`` at publish time, no extra
    revision).

Deliberately out of scope: batch CLI, multi-chunk iteration, A4/A5/A6, H3.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from short_drama.artifacts import ArtifactRef, FileArtifactStore, canonical_json_bytes
from short_drama.foundation import (
    FilePointerStore,
    VALIDATION_REPORT_ARTIFACT_TYPE,
    ValidationResult,
    load_validation_report,
)
from short_drama.llm import (
    LLMClient,
    LLMConfigError,
    LLMRetryExhaustedError,
    LLMStructuredOutputError,
    LLMTransportError,
    LLMInvocationProvenance,
    OutputSchema,
    ProviderMeta,
    PromptRegistry,
    PromptSpec,
    ReasoningSettings,
    SemanticLLMProfile,
    StructuredGenerationResult,
    build_provenance,
    build_structured_request,
    validate_against_output_schema,
)
from short_drama.paths import REPO_ROOT, SCHEMAS_DIR
from short_drama.story import (
    CandidateExtractionPublication,
    ChunkExtractionService,
    DEFAULT_OUTPUT_SCHEMA_PATH,
    ExtractionModelError,
    ExtractionProvenanceError,
    ExtractionSemanticGenerationError,
    LANGUAGE_DETECTOR_ID,
    SourceChapter,
    SourceChunk,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    ParagraphSpan,
    StoryConfigError,
    StoryExtractionProfile,
    StoryIntegrityError,
    TOKEN_COUNTER_ID,
    UnresolvedMentionCandidate,
    build_chunk_context,
    load_candidate_extraction,
    persist_source_chunk,
    persist_source_document,
)
from short_drama.story.extraction import (
    CandidatePayload,
    CharacterCandidate,
    EventCandidate,
    EvidenceRef,
    FactCandidate,
)
from short_drama.story.persistence import (
    source_chunk_artifact_id,
    source_document_artifact_id,
)
from short_drama.story.source import NormalizationInfo

# ---------------------------------------------------------------------------
# Frozen identity constants (mirror the A3C test fixture)
# ---------------------------------------------------------------------------

PROJECT = "classroom"
DOCUMENT = "src_001"
CHUNK_PROFILE_ID = "story-analysis-v1"
CHUNK_ID = "CH001_C001"
CHAPTER_ID = "CH001"
EXTRACTION_PROFILE_ID = "story-extraction-v1"
PROMPTS_DIR = REPO_ROOT / "prompts" / "story"

PARAGRAPHS = {
    "CH001_P0001": "左上下文第一段。",
    "CH001_P0002": "左上下文第二段。",
    "CH001_P0003": "林晚走进教室。",
    "CH001_P0004": "老师正在板书。",
    "CH001_P0005": "右上下文第一段。",
    "CH001_P0006": "右上下文第二段。",
}
OWNERSHIP = ("CH001_P0003", "CH001_P0004")


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def make_evidence(
    paragraph_id: str = "CH001_P0003",
    role: str = "primary",
    strength: str = "explicit",
    excerpt: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role=role, strength=strength, excerpt=excerpt
    )


def make_character(candidate_id: str, display_name: str, evidence) -> CharacterCandidate:
    return CharacterCandidate(
        candidate_id=candidate_id,
        display_name_original=display_name,
        aliases_original=(),
        descriptors_zh=(),
        summary_zh=f"{display_name}的简要描述。",
        evidence_strength="explicit",
        evidence=tuple(evidence),
    )


def canonical_payload() -> CandidatePayload:
    """A valid, already-canonical candidate payload (primary evidence in
    ownership span)."""
    return CandidatePayload(
        characters=(
            make_character("cand_char_001", "林晚", (make_evidence("CH001_P0003"),)),
            make_character("cand_char_002", "老师", (make_evidence("CH001_P0004"),)),
        ),
    )


def invalid_payload() -> CandidatePayload:
    """JSON-Schema-valid but A3B-semantic-invalid (dangling local cross-ref)."""
    return CandidatePayload(
        facts=(
            FactCandidate(
                candidate_id="cand_fact_001",
                fact_type="identity",
                statement_zh="林晚是老李的女儿。",
                subject_refs=("cand_char_999",),  # dangling local ref
                object_refs=(),
                evidence_strength="explicit",
                evidence=(make_evidence("CH001_P0003"),),
            ),
        ),
    )


def mismatched_excerpt_payload() -> CandidatePayload:
    """Otherwise-valid payload with ONE mismatched excerpt + one exact excerpt.

    ``cand_char_001``'s excerpt is NOT a substring of its paragraph, so the
    Issue #37 sanitizer nulls it; ``cand_char_002``'s excerpt IS an exact
    substring and is preserved byte-for-byte. After sanitization the payload
    passes A3B on semantic round 1 (no second generation call).
    """
    return CandidatePayload(
        characters=(
            make_character(
                "cand_char_001",
                "林晚",
                (make_evidence("CH001_P0003", excerpt="不存在的文字"),),
            ),
            make_character(
                "cand_char_002",
                "老师",
                (make_evidence("CH001_P0004", excerpt="老师正在"),),
            ),
        ),
    )


def non_excerpt_failure_payload() -> CandidatePayload:
    """Genuine non-excerpt semantic failure (dangling local ref) + one
    mismatched excerpt (which the sanitizer nulls away).

    After sanitization the ONLY remaining finding is the dangling ref
    (``A3_LOCAL_REF_NOT_FOUND``) -- no ``A3_EVIDENCE_EXCERPT_MISMATCH`` -- so
    the existing bounded semantic-regeneration behavior still applies.
    """
    return CandidatePayload(
        characters=(
            make_character(
                "cand_char_001",
                "林晚",
                (make_evidence("CH001_P0003", excerpt="不存在的文字"),),
            ),
        ),
        facts=(
            FactCandidate(
                candidate_id="cand_fact_001",
                fact_type="identity",
                statement_zh="林晚是老李的女儿。",
                subject_refs=("cand_char_999",),  # dangling local ref
                object_refs=(),
                evidence_strength="explicit",
                evidence=(make_evidence("CH001_P0003"),),
            ),
        ),
    )


def uncertainty_payload() -> CandidatePayload:
    """A legitimate unresolved/uncertain payload that PASSES A3B (no retry)."""
    return CandidatePayload(
        characters=(
            make_character("cand_char_001", "林晚", (make_evidence("CH001_P0003"),)),
        ),
        events=(
            EventCandidate(
                candidate_id="cand_evt_001",
                summary_zh="林晚走进教室。",
                participant_refs=("cand_char_001",),
                location_refs=(),
                temporal_mode="unknown",
                evidence_strength="uncertain",
                evidence=(make_evidence("CH001_P0003"),),
            ),
        ),
        unresolved_mentions=(
            UnresolvedMentionCandidate(
                candidate_id="cand_unres_001",
                mention_original="她",
                mention_kind="person",
                reason_zh="代词指代不确定。",
                possible_candidate_refs=(),  # empty is valid
                evidence_strength="uncertain",
                evidence=(make_evidence("CH001_P0004"),),
            ),
        ),
    )


def typed_reject_payload_dict() -> dict:
    """A CandidatePayload JSON-Schema-valid object the A3 typed model rejects.

    The provider schema for ``characters[].aliases_original`` is
    ``{"type": "array", "items": {"type": "string", "minLength": 1}}`` with NO
    ``uniqueItems`` constraint, so ``["小晚", "小晚"]`` passes A-I3 local JSON
    Schema validation. But the A3A typed model rejects duplicate strings in
    ``aliases_original`` (``_to_string_tuple(..., unique=True)``), so
    ``CandidatePayload.from_dict`` raises ``ExtractionModelError``. The other
    fields are valid, so the ONLY failure is the typed uniqueness rule. No A3B
    ValidationFindings are produced (semantic validation never runs for a
    payload that fails typed loading).
    """
    return {
        "characters": [
            {
                "candidate_id": "cand_char_001",
                "display_name_original": "林晚",
                "aliases_original": ["小晚", "小晚"],  # schema-valid, typed-model-invalid
                "descriptors_zh": [],
                "summary_zh": "林晚的简要描述。",
                "evidence_strength": "explicit",
                "evidence": [
                    {
                        "paragraph_id": "CH001_P0003",
                        "role": "primary",
                        "strength": "explicit",
                        "excerpt": None,
                    }
                ],
            }
        ],
        "locations": [],
        "facts": [],
        "events": [],
        "relationships": [],
        "unresolved_mentions": [],
    }


def _dict(payload: CandidatePayload) -> dict:
    return payload.to_dict()


# ---------------------------------------------------------------------------
# Source / profile builders
# ---------------------------------------------------------------------------


def build_source_document(paragraphs: dict[str, str] | None = None) -> SourceDocument:
    paragraphs = paragraphs or PARAGRAPHS
    chapter = SourceChapter(
        CHAPTER_ID,
        None,
        "synthetic",
        tuple(SourceParagraph(pid, paragraphs[pid], None) for pid in sorted(paragraphs)),
    )
    return SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt", "source/novel.txt", "f" * 64, 128, "zh-CN", "zh", LANGUAGE_DETECTOR_ID
        ),
        normalization=NormalizationInfo("utf-8", "LF", "short_drama_source_ingestion_v1", "1"),
        chapters=(chapter,),
    )


def build_source_chunk(
    source_document_ref: ArtifactRef,
    pids: tuple[str, ...] | None = None,
    ownership: tuple[str, str] | None = None,
) -> SourceChunk:
    pids = pids or tuple(sorted(PARAGRAPHS))
    ownership = ownership or OWNERSHIP
    return SourceChunk(
        schema_version=1,
        chunk_id=CHUNK_ID,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chapter_id=CHAPTER_ID,
        source_document_ref=source_document_ref,
        context_span=ParagraphSpan(pids[0], pids[-1]),
        ownership_span=ParagraphSpan(ownership[0], ownership[1]),
        paragraph_ids=pids,
        token_count_method=TOKEN_COUNTER_ID,
        context_token_count=10,
        ownership_token_count=5,
    )


def make_extraction_profile(**overrides) -> StoryExtractionProfile:
    values = {
        "schema_version": 1,
        "profile_id": EXTRACTION_PROFILE_ID,
        "working_language": "zh-CN",
        "prompt_id": "a3.chunk-extraction",
        "prompt_version": 1,
        "output_schema_id": "a3-candidate-payload",
        "output_schema_version": 1,
        "max_generation_rounds": 2,
    }
    values.update(overrides)
    return StoryExtractionProfile(**values)


def make_semantic_profile(**overrides) -> SemanticLLMProfile:
    # A semantic profile carries NO backend identity (no provider_family and no
    # model); the backend is supplied by the RuntimeConfig / build_provenance.
    values = {
        "schema_version": 2,
        "profile_id": "story-extraction-llm-v1",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "structured_output_mode": "json_schema",
        "reasoning": ReasoningSettings(enabled=False),
    }
    values.update(overrides)
    return SemanticLLMProfile(**values)


# ---------------------------------------------------------------------------
# Fake provider-neutral LLM client
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """Deterministic fake provider. Returns a scripted sequence of results.

    Each scripted element is one of:
      * a ``dict`` -> a successful ``StructuredGenerationResult`` whose
        provenance is built from the request (matching), via
        ``build_provenance``;
      * a ``(dict, LLMInvocationProvenance)`` tuple -> a successful result with
        the EXACT given provenance (used to fabricate a provenance mismatch);
      * an ``Exception`` instance -> raised from ``generate_structured``.

    The fake honors the real ``LLMClient.generate_structured()`` trust
    boundary: every scripted payload that would be returned as a *successful*
    result is first passed through the existing A-I3
    :func:`validate_against_output_schema` against the request's output schema.
    A schema-invalid payload therefore never escapes as a successful result (it
    raises ``LLMStructuredOutputError``), so a successful result is always
    JSON-Schema-valid and the only remaining rejection path is the A3 typed
    domain load.
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.calls = []  # (rendered_prompt, output_schema, semantic_profile)
        self.request_hashes = []
        self.user_texts = []
        self.pre_call_hook = None

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
        self.calls.append((rendered_prompt, output_schema, semantic_profile))
        self.user_texts.append(rendered_prompt.user_text)
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        self.request_hashes.append(request.request_hash)
        if self.pre_call_hook is not None:
            self.pre_call_hook(rendered_prompt, output_schema, semantic_profile, request)
        if not self.responses:
            raise AssertionError("unexpected extra generate_structured call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            parsed, provenance = response
        else:
            parsed = response
            provenance = build_provenance(
                request, ProviderMeta(), request_model="qwen3-27b", provider_family="qwen"
            )
        # Trust boundary: a successful result must have passed strict local
        # JSON Schema validation against the request's output schema.
        validate_against_output_schema(parsed, output_schema)
        return StructuredGenerationResult(
            parsed_json=parsed, provenance=provenance, attempts=1
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class Harness:
    store: FileArtifactStore
    pointers: FilePointerStore
    service: ChunkExtractionService
    source_document: SourceDocument
    source_document_ref: ArtifactRef
    source_chunk: SourceChunk
    source_chunk_ref: ArtifactRef
    profile: StoryExtractionProfile
    semantic_profile: SemanticLLMProfile


def make_harness(
    root: Path,
    *,
    source_document: SourceDocument | None = None,
    source_chunk: SourceChunk | None = None,
    profile: StoryExtractionProfile | None = None,
    semantic_profile: SemanticLLMProfile | None = None,
) -> Harness:
    store = FileArtifactStore(root / "artifacts")
    pointers = FilePointerStore(root / "pointers", store)
    source_document = source_document or build_source_document()
    source_document_ref = persist_source_document(store, source_document, revision=1)
    source_chunk = source_chunk or build_source_chunk(source_document_ref)
    source_chunk_ref = persist_source_chunk(
        store, source_chunk, profile_id=CHUNK_PROFILE_ID, revision=1
    )
    profile = profile or make_extraction_profile()
    semantic_profile = semantic_profile or make_semantic_profile()
    service = ChunkExtractionService(
        store, pointers, PromptRegistry(PROMPTS_DIR), DEFAULT_OUTPUT_SCHEMA_PATH
    )
    return Harness(
        store=store,
        pointers=pointers,
        service=service,
        source_document=source_document,
        source_document_ref=source_document_ref,
        source_chunk=source_chunk,
        source_chunk_ref=source_chunk_ref,
        profile=profile,
        semantic_profile=semantic_profile,
    )


def run(h: Harness, client: FakeLLMClient, **overrides) -> CandidateExtractionPublication:
    return h.service.extract_chunk(
        source_document_ref=overrides.get("source_document_ref", h.source_document_ref),
        source_chunk_ref=overrides.get("source_chunk_ref", h.source_chunk_ref),
        chunk_profile_id=overrides.get("chunk_profile_id", CHUNK_PROFILE_ID),
        extraction_profile=overrides.get("extraction_profile", h.profile),
        semantic_profile=overrides.get("semantic_profile", h.semantic_profile),
        llm_client=client,
    )


def store_path(
    store: FileArtifactStore, artifact_type: str, artifact_id: str, revision: int
) -> Path:
    return store.root / artifact_type / artifact_id / f"r{revision:08d}.json"


def _extraction_artifact_id() -> str:
    from short_drama.story import candidate_extraction_artifact_id

    return candidate_extraction_artifact_id(
        PROJECT, DOCUMENT, CHUNK_PROFILE_ID, CHUNK_ID, EXTRACTION_PROFILE_ID
    )


def _validation_artifact_id() -> str:
    from short_drama.story import candidate_extraction_validation_artifact_id

    return candidate_extraction_validation_artifact_id(_extraction_artifact_id())


def _current_target_ref(h: Harness) -> ArtifactRef | None:
    from short_drama.story import candidate_extraction_pointer_id

    pointer_id = candidate_extraction_pointer_id(
        PROJECT, DOCUMENT, CHUNK_PROFILE_ID, CHUNK_ID, EXTRACTION_PROFILE_ID
    )
    try:
        return h.pointers.resolve_current(pointer_id).target_ref
    except Exception:  # noqa: BLE001
        return None


def _alt_source_pair(h: Harness):
    """A coherent alternate source pair: a different doc content (revision 2) +
    a chunk pinning exactly that doc ref (revision 2)."""
    alt_doc = build_source_document(
        paragraphs={
            "CH001_P0001": "不同的左上下文第一段。",
            "CH001_P0002": "不同的左上下文第二段。",
            "CH001_P0003": "不同的林晚走进教室。",
            "CH001_P0004": "不同的老师正在板书。",
            "CH001_P0005": "不同的右上下文第一段。",
            "CH001_P0006": "不同的右上下文第二段。",
        }
    )
    alt_doc_ref = persist_source_document(h.store, alt_doc, revision=2)
    alt_chunk = build_source_chunk(alt_doc_ref)
    alt_chunk_ref = persist_source_chunk(
        h.store, alt_chunk, profile_id=CHUNK_PROFILE_ID, revision=2
    )
    return alt_doc, alt_doc_ref, alt_chunk, alt_chunk_ref


# ===========================================================================
# Context / rendering
# ===========================================================================


def test_deterministic_partition():
    doc = build_source_document()
    chunk = build_source_chunk(
        ArtifactRef("source_document", "classroom.src_001", 1, "f" * 64)
    )
    from short_drama.story import partition_paragraph_ids

    left, ownership, right = partition_paragraph_ids(chunk)
    assert left == ["CH001_P0001", "CH001_P0002"]
    assert ownership == ["CH001_P0003", "CH001_P0004"]
    assert right == ["CH001_P0005", "CH001_P0006"]


def test_paragraph_order_preserved():
    doc = build_source_document()
    chunk = build_source_chunk(
        ArtifactRef("source_document", "classroom.src_001", 1, "f" * 64)
    )
    context = build_chunk_context(doc, chunk)
    left = json.loads(context.left_context_json)
    assert [item["paragraph_id"] for item in left] == ["CH001_P0001", "CH001_P0002"]
    assert left[0]["text_original"] == "左上下文第一段。"
    assert left[1]["text_original"] == "左上下文第二段。"


def test_canonical_paragraph_json_stable():
    doc = build_source_document()
    chunk = build_source_chunk(
        ArtifactRef("source_document", "classroom.src_001", 1, "f" * 64)
    )
    context_a = build_chunk_context(doc, chunk)
    context_b = build_chunk_context(doc, chunk)
    assert context_a.ownership_json == context_b.ownership_json
    # the rendered string must BE the canonical (RFC 8785) form: re-canonicalizing
    # the parsed value reproduces the exact bytes.
    parsed = json.loads(context_a.ownership_json)
    assert canonical_json_bytes(parsed).decode("utf-8") == context_a.ownership_json


def test_empty_left_accepted():
    doc = build_source_document()
    chunk = build_source_chunk(
        ArtifactRef("source_document", "classroom.src_001", 1, "f" * 64),
        ownership=("CH001_P0001", "CH001_P0002"),
    )
    context = build_chunk_context(doc, chunk)
    assert context.left_context_json == "[]"
    assert json.loads(context.ownership_json) and len(json.loads(context.ownership_json)) == 2


def test_empty_right_accepted():
    doc = build_source_document()
    chunk = build_source_chunk(
        ArtifactRef("source_document", "classroom.src_001", 1, "f" * 64),
        ownership=("CH001_P0005", "CH001_P0006"),
    )
    context = build_chunk_context(doc, chunk)
    assert context.right_context_json == "[]"


def test_ownership_non_empty():
    doc = build_source_document()
    chunk = build_source_chunk(
        ArtifactRef("source_document", "classroom.src_001", 1, "f" * 64),
        ownership=("CH001_P0003", "CH001_P0003"),
    )
    from short_drama.story import partition_paragraph_ids

    _left, ownership, _right = partition_paragraph_ids(chunk)
    assert len(ownership) >= 1
    assert ownership == ["CH001_P0003"]


def test_ownership_endpoint_missing_from_source_document_fails_before_llm(tmp_path):
    # The document only has P0001..P0003, so the default chunk's ownership END
    # (P0004) does not resolve in the exact SourceDocument.
    doc = build_source_document(
        paragraphs={pid: PARAGRAPHS[pid] for pid in ("CH001_P0001", "CH001_P0002", "CH001_P0003")}
    )
    h = make_harness(tmp_path, source_document=doc)
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryIntegrityError):
        run(h, client)
    assert client.call_count == 0


def test_missing_paragraph_fails_before_llm(tmp_path):
    # The document only has P0001..P0005, so a RIGHT-context paragraph (P0006)
    # does not resolve in the exact SourceDocument.
    doc = build_source_document(
        paragraphs={pid: PARAGRAPHS[pid] for pid in ("CH001_P0001", "CH001_P0002", "CH001_P0003", "CH001_P0004", "CH001_P0005")}
    )
    h = make_harness(tmp_path, source_document=doc)
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryIntegrityError):
        run(h, client)
    assert client.call_count == 0


def test_source_chunk_lineage_mismatch_fails_before_llm(tmp_path):
    # A chunk that pins a DIFFERENT source document ref than the requested one.
    h = make_harness(tmp_path)
    _alt_doc, alt_doc_ref, alt_chunk, alt_chunk_ref = _alt_source_pair(h)
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryIntegrityError):
        run(
            h,
            client,
            source_document_ref=h.source_document_ref,
            source_chunk_ref=alt_chunk_ref,
        )
    assert client.call_count == 0


def test_source_text_with_template_braces_does_not_become_template_syntax(tmp_path):
    doc = build_source_document(
        paragraphs={
            "CH001_P0001": "左上下文{{chunk_id}}第一段。",
            "CH001_P0002": "左上下文{{ownership_json}}第二段。",
            "CH001_P0003": "林晚走进教室。",
            "CH001_P0004": "老师正在板书。",
            "CH001_P0005": "右上下文第一段。",
            "CH001_P0006": "右上下文第二段。",
        }
    )
    h = make_harness(tmp_path, source_document=doc)
    client = FakeLLMClient([_dict(canonical_payload())])
    run(h, client)
    assert client.call_count == 1
    # The literal template-like braces from the source text survive rendering as
    # literal text (not substituted template syntax).
    assert "{{chunk_id}}" in client.user_texts[0]
    assert "{{ownership_json}}" in client.user_texts[0]


# ===========================================================================
# Profile / prompt / schema consistency
# ===========================================================================


def _mismatched_prompt_spec(prompt_id: str = "a3.chunk-extraction", version: int = 1) -> PromptSpec:
    return PromptSpec.create(
        prompt_id=prompt_id,
        version=version,
        system_template="You extract story candidates.",
        user_template="chunk_id = {{chunk_id}}",
        required_variables=["chunk_id"],
    )


def test_prompt_id_mismatch_fails_before_llm(tmp_path, monkeypatch):
    h = make_harness(tmp_path)
    monkeypatch.setattr(
        h.service, "_load_prompt_spec", lambda profile: _mismatched_prompt_spec(prompt_id="a3.other")
    )
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryConfigError, match="prompt_id"):
        run(h, client)
    assert client.call_count == 0


def test_prompt_version_mismatch_fails_before_llm(tmp_path, monkeypatch):
    h = make_harness(tmp_path)
    monkeypatch.setattr(
        h.service, "_load_prompt_spec", lambda profile: _mismatched_prompt_spec(version=2)
    )
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryConfigError, match="prompt_version"):
        run(h, client)
    assert client.call_count == 0


def test_output_schema_id_mismatch_fails_before_llm(tmp_path, monkeypatch):
    h = make_harness(tmp_path)
    monkeypatch.setattr(
        h.service,
        "_build_output_schema",
        lambda profile: OutputSchema.create(
            schema_id="a3.other-schema", schema_version=1, schema={"type": "object"}
        ),
    )
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryConfigError, match="output_schema_id"):
        run(h, client)
    assert client.call_count == 0


def test_output_schema_version_mismatch_fails_before_llm(tmp_path, monkeypatch):
    h = make_harness(tmp_path)
    monkeypatch.setattr(
        h.service,
        "_build_output_schema",
        lambda profile: OutputSchema.create(
            schema_id="a3-candidate-payload", schema_version=2, schema={"type": "object"}
        ),
    )
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryConfigError, match="output_schema_version"):
        run(h, client)
    assert client.call_count == 0


def test_valid_profile_prompt_schema_builds_stable_request_hash(tmp_path):
    h1 = make_harness(tmp_path / "a")
    h2 = make_harness(tmp_path / "b")
    c1 = FakeLLMClient([_dict(canonical_payload())])
    c2 = FakeLLMClient([_dict(canonical_payload())])
    run(h1, c1)
    run(h2, c2)
    assert c1.call_count == 1 and c2.call_count == 1
    assert c1.request_hashes[0] == c2.request_hashes[0]
    # and it matches the request built independently from the same inputs
    registry = PromptRegistry(PROMPTS_DIR)
    spec = registry.load("a3.chunk-extraction", version=1)
    from short_drama.llm import render_prompt

    rendered = render_prompt(spec, build_chunk_context(h1.source_document, h1.source_chunk).as_variables())
    schema = h1.service._build_output_schema(h1.profile)
    request = build_structured_request(
        rendered_prompt=rendered, output_schema=schema, semantic_profile=h1.semantic_profile
    )
    assert request.request_hash == c1.request_hashes[0]


# ===========================================================================
# Reuse
# ===========================================================================


def test_existing_current_reuses_with_zero_provider_calls(tmp_path):
    h = make_harness(tmp_path)
    first = run(h, FakeLLMClient([_dict(canonical_payload())]))
    assert first.reused is False
    second_client = FakeLLMClient([_dict(canonical_payload())])
    second = run(h, second_client)
    assert second.reused is True
    assert second_client.call_count == 0
    assert second.candidate_extraction_ref == first.candidate_extraction_ref
    assert second.validation_report_ref == first.validation_report_ref
    assert second.current_pointer_ref == first.current_pointer_ref


def test_stale_semantic_identity_provider_called(tmp_path):
    h = make_harness(tmp_path)
    # Publish a CURRENT under a DIFFERENT semantic identity (temperature 0.5).
    run(h, FakeLLMClient([_dict(canonical_payload())]), semantic_profile=make_semantic_profile(temperature=0.5))
    # Request the original identity (temperature 0.0): normal miss -> generate.
    client = FakeLLMClient([_dict(canonical_payload())])
    result = run(h, client)
    assert client.call_count == 1
    assert result.reused is False


def test_corrupt_current_fails_closed_zero_provider_calls(tmp_path):
    h = make_harness(tmp_path)
    run(h, FakeLLMClient([_dict(canonical_payload())]))
    # Corrupt the CURRENT by removing its exact ValidationReport.
    store_path(h.store, VALIDATION_REPORT_ARTIFACT_TYPE, _validation_artifact_id(), 1).unlink()
    client = FakeLLMClient([_dict(canonical_payload())])
    with pytest.raises(StoryIntegrityError):
        run(h, client)
    assert client.call_count == 0


# ===========================================================================
# Successful generation
# ===========================================================================


def test_first_valid_result_one_round_and_publish(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(canonical_payload())])
    result = run(h, client)
    assert client.call_count == 1
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 1
    # A3C typed load round-trips to the exact canonical payload.
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert loaded.candidates == canonical_payload()


def test_exact_validation_report_and_current(tmp_path):
    h = make_harness(tmp_path)
    result = run(h, FakeLLMClient([_dict(canonical_payload())]))
    report = load_validation_report(h.store, result.validation_report_ref)
    assert {(ref.role, ref.artifact_ref) for ref in report.validated_refs} == {
        ("source_document", h.source_document_ref),
        ("source_chunk", h.source_chunk_ref),
        ("candidate_extraction", result.candidate_extraction_ref),
    }
    assert report.summary.result is ValidationResult.PASS
    assert _current_target_ref(h) == result.candidate_extraction_ref


# ===========================================================================
# Semantic regeneration
# ===========================================================================


def test_invalid_round1_valid_round2_success(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(invalid_payload()), _dict(canonical_payload())])
    result = run(h, client)
    assert client.call_count == 2
    assert result.reused is False
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert loaded.candidates == canonical_payload()


def test_both_rounds_use_exact_same_request_hash(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(invalid_payload()), _dict(canonical_payload())])
    run(h, client)
    assert client.request_hashes[0] == client.request_hashes[1]
    # no adaptive repair prompt: the rendered user text is byte-identical
    assert client.user_texts[0] == client.user_texts[1]


def test_exactly_two_generate_calls_on_both_invalid(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(invalid_payload()), _dict(invalid_payload())])
    with pytest.raises(ExtractionSemanticGenerationError):
        run(h, client)
    assert client.call_count == 2


def test_semantic_exhaustion_bounded_failure_diagnostics(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(invalid_payload()), _dict(invalid_payload())])
    with pytest.raises(ExtractionSemanticGenerationError) as exc_info:
        run(h, client)
    err = exc_info.value
    assert err.rounds_attempted == 2
    assert len(err.final_findings) >= 1
    assert err.final_validation_result is not None
    assert err.final_validation_result.result is ValidationResult.FAIL
    assert "A3_LOCAL_REF_NOT_FOUND" in {f.code for f in err.final_findings}


def test_no_extraction_persisted_for_invalid_payloads(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(invalid_payload()), _dict(invalid_payload())])
    with pytest.raises(ExtractionSemanticGenerationError):
        run(h, client)
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", _extraction_artifact_id(), 1)
    assert _current_target_ref(h) is None


def test_previous_valid_current_preserved_on_exhaustion(tmp_path):
    h = make_harness(tmp_path)
    first = run(h, FakeLLMClient([_dict(canonical_payload())]))
    # A DIFFERENT semantic identity, both rounds invalid -> exhaustion. The
    # prior valid CURRENT (original identity) must remain unchanged.
    client = FakeLLMClient([_dict(invalid_payload()), _dict(invalid_payload())])
    with pytest.raises(ExtractionSemanticGenerationError):
        run(h, client, semantic_profile=make_semantic_profile(temperature=0.5))
    assert _current_target_ref(h) == first.candidate_extraction_ref
    # no new revision was manufactured for the exhausted identity
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", _extraction_artifact_id(), 2)


def test_typed_reject_fixture_is_schema_valid_but_typed_invalid(tmp_path):
    # Directly prove the fixture is genuinely JSON-Schema-valid yet rejected by
    # the A3A typed model (the ONLY failing case for this round).
    h = make_harness(tmp_path)
    schema = h.service._build_output_schema(h.profile)
    # 1. A-I3 local JSON Schema validation PASSES (no LLMStructuredOutputError).
    validate_against_output_schema(typed_reject_payload_dict(), schema)
    # 2. The A3A typed domain model REJECTS it (duplicate aliases_original).
    with pytest.raises(ExtractionModelError):
        CandidatePayload.from_dict(typed_reject_payload_dict())


def test_fake_client_honors_local_schema_validation(tmp_path):
    # A payload that FAILS the provider schema must not escape the fake as a
    # successful result: the fake applies validate_against_output_schema and
    # raises LLMStructuredOutputError (mirroring the real LLMClient trust
    # boundary). The schema-invalid payload never reaches A3 typed loading.
    h = make_harness(tmp_path)
    bad = _dict(canonical_payload())
    bad["characters"][0]["evidence_strength"] = "bogus"  # enum violation
    client = FakeLLMClient([bad])
    with pytest.raises(LLMStructuredOutputError):
        run(h, client)
    assert client.call_count == 1
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", _extraction_artifact_id(), 1)


def test_typed_domain_rejection_consumes_a_round(tmp_path):
    # round 1: JSON-Schema-valid but typed-domain-invalid (no A3B findings);
    # round 2: valid -> publish. Proves a typed-model rejection consumes a
    # semantic round without persisting anything for it.
    h = make_harness(tmp_path)
    client = FakeLLMClient([typed_reject_payload_dict(), _dict(canonical_payload())])
    result = run(h, client)
    assert client.call_count == 2
    assert result.reused is False
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert loaded.candidates == canonical_payload()


def test_typed_domain_rejection_twice_exhausts_without_findings(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([typed_reject_payload_dict(), typed_reject_payload_dict()])
    with pytest.raises(ExtractionSemanticGenerationError) as exc_info:
        run(h, client)
    assert client.call_count == 2
    assert exc_info.value.rounds_attempted == 2
    # a typed-domain rejection produces no A3B ValidationFindings / result
    assert exc_info.value.final_findings == ()
    assert exc_info.value.final_validation_result is None
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", _extraction_artifact_id(), 1)
    assert _current_target_ref(h) is None


# ===========================================================================
# Issue #37: deterministic excerpt sanitization before semantic validation
# ===========================================================================


def test_one_mismatched_excerpt_sanitizes_and_passes_on_round1(tmp_path):
    # An otherwise-valid payload whose only defect is one mismatched excerpt
    # sanitizes (excerpt -> None) and passes on semantic round 1 with NO second
    # generation call.
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(mismatched_excerpt_payload())])
    result = run(h, client)
    assert client.call_count == 1
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 1


def test_published_extraction_contains_null_for_sanitized_excerpt(tmp_path):
    # Only the sanitized payload is published: the mismatched excerpt is
    # persisted as ``null``; the exact excerpt is preserved byte-for-byte. The
    # inaccurate source quote is never persisted.
    h = make_harness(tmp_path)
    run(h, FakeLLMClient([_dict(mismatched_excerpt_payload())]))
    ref = _current_target_ref(h)
    loaded = load_candidate_extraction(h.store, ref)
    by_id = {
        c.candidate_id: c for c in loaded.candidates.characters
    }
    assert by_id["cand_char_001"].evidence[0].excerpt is None
    assert by_id["cand_char_002"].evidence[0].excerpt == "老师正在"
    # and the persisted payload still re-validates to PASS
    from short_drama.story import validate_candidate_payload

    assert validate_candidate_payload(
        loaded.candidates, h.source_document, h.source_chunk
    ).is_valid is True


def test_exact_rerun_reuses_sanitized_validated_current(tmp_path):
    # An exact rerun of a chunk whose CURRENT was produced by sanitizing a
    # mismatched excerpt reuses that validated CURRENT with zero provider calls.
    h = make_harness(tmp_path)
    first = run(h, FakeLLMClient([_dict(mismatched_excerpt_payload())]))
    assert first.reused is False
    second_client = FakeLLMClient([_dict(mismatched_excerpt_payload())])
    second = run(h, second_client)
    assert second.reused is True
    assert second_client.call_count == 0
    assert second.candidate_extraction_ref == first.candidate_extraction_ref
    loaded = load_candidate_extraction(h.store, second.candidate_extraction_ref)
    by_id = {c.candidate_id: c for c in loaded.candidates.characters}
    assert by_id["cand_char_001"].evidence[0].excerpt is None


def test_non_excerpt_failure_still_bounded_regeneration(tmp_path):
    # A genuine non-excerpt semantic failure (dangling local ref) still drives
    # the existing bounded regeneration: both rounds consume a call and exhaust,
    # and the sanitizer removed the excerpt defect (no excerpt-mismatch finding).
    h = make_harness(tmp_path)
    client = FakeLLMClient(
        [_dict(non_excerpt_failure_payload()), _dict(non_excerpt_failure_payload())]
    )
    with pytest.raises(ExtractionSemanticGenerationError) as exc_info:
        run(h, client)
    assert client.call_count == 2
    codes = {f.code for f in exc_info.value.final_findings}
    assert "A3_LOCAL_REF_NOT_FOUND" in codes
    assert "A3_EVIDENCE_EXCERPT_MISMATCH" not in codes


# ===========================================================================
# Uncertainty is success
# ===========================================================================


def test_unresolved_uncertain_payload_no_retry(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(uncertainty_payload())])
    result = run(h, client)
    assert client.call_count == 1
    assert result.reused is False
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert len(loaded.candidates.unresolved_mentions) == 1
    assert loaded.candidates.unresolved_mentions[0].evidence_strength == "uncertain"


# ===========================================================================
# A-I3 retry delegation
# ===========================================================================


def test_llm_retry_exhausted_propagated_no_second_semantic_round(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([LLMRetryExhaustedError(attempts=3, message="exhausted")])
    with pytest.raises(LLMRetryExhaustedError):
        run(h, client)
    # A-I3 already exhausted its technical budget; A3D must NOT start a second
    # semantic round.
    assert client.call_count == 1


def test_non_retryable_llm_config_error_propagated_unchanged(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([LLMConfigError("adapter does not support mode")])
    with pytest.raises(LLMConfigError, match="adapter does not support"):
        run(h, client)
    assert client.call_count == 1


def test_no_nested_technical_retry_in_a3d(tmp_path):
    # A3D must not wrap the provider in its own technical retry loop. A
    # retryable technical error raised by generate_structured (which A-I3 would
    # retry *internally*) is, from A3D's perspective, propagated after exactly
    # ONE provider call — no second provider round is started by A3D.
    h = make_harness(tmp_path)
    client = FakeLLMClient([LLMTransportError("transient connection failure")])
    with pytest.raises(LLMTransportError):
        run(h, client)
    assert client.call_count == 1


# ===========================================================================
# Provenance
# ===========================================================================


def test_matching_provenance_succeeds(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_dict(canonical_payload())])
    result = run(h, client)
    assert result.reused is False


def test_mismatched_provenance_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    # Fabricate a result whose provenance carries a DIFFERENT request_hash.
    valid = _dict(canonical_payload())
    client = FakeLLMClient([_mismatched_provenance_result(h, valid)])
    with pytest.raises(ExtractionProvenanceError):
        run(h, client)
    assert client.call_count == 1


def test_mismatched_provenance_never_persisted(tmp_path):
    h = make_harness(tmp_path)
    client = FakeLLMClient([_mismatched_provenance_result(h, _dict(canonical_payload()))])
    with pytest.raises(ExtractionProvenanceError):
        run(h, client)
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", _extraction_artifact_id(), 1)
    assert _current_target_ref(h) is None


def _mismatched_provenance_result(h: Harness, parsed: dict):
    """Build a (parsed, provenance) tuple whose provenance request_hash is
    deliberately wrong."""
    registry = PromptRegistry(PROMPTS_DIR)
    spec = registry.load("a3.chunk-extraction", version=1)
    from short_drama.llm import render_prompt

    rendered = render_prompt(spec, build_chunk_context(h.source_document, h.source_chunk).as_variables())
    schema = h.service._build_output_schema(h.profile)
    request = build_structured_request(
        rendered_prompt=rendered, output_schema=schema, semantic_profile=h.semantic_profile
    )
    provenance = build_provenance(
        request, ProviderMeta(), request_model="qwen3-27b", provider_family="qwen"
    )
    return (parsed, dataclasses.replace(provenance, request_hash="0" * 64))


# ===========================================================================
# Publish race (same-identity A3C reuse at publish time)
# ===========================================================================


def test_publish_time_same_identity_reuses_no_extra_revision(tmp_path):
    h = make_harness(tmp_path)

    def competing_publish(rendered_prompt, output_schema, semantic_profile, request):
        # Another worker publishes the SAME semantic identity during our provider
        # call window, before our own publish_validated runs.
        h.service._persistence.publish_validated(
            source_document=h.source_document,
            source_document_ref=h.source_document_ref,
            source_chunk=h.source_chunk,
            source_chunk_ref=h.source_chunk_ref,
            chunk_profile_id=CHUNK_PROFILE_ID,
            extraction_profile=h.profile,
            generation_provenance=build_provenance(
                request, ProviderMeta(), request_model="qwen3-27b", provider_family="qwen"
            ),
            payload=canonical_payload(),
        )

    client = FakeLLMClient([_dict(canonical_payload())])
    client.pre_call_hook = competing_publish
    result = run(h, client)
    assert client.call_count == 1
    assert result.reused is True
    assert result.candidate_extraction_ref.revision == 1
    # no second revision was manufactured
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", _extraction_artifact_id(), 2)
    assert _current_target_ref(h) == result.candidate_extraction_ref
