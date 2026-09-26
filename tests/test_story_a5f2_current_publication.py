"""Tests for v1.2 A5F2: validated publication + upstream A4 stability + CAS CURRENT.

Covers:
  * the A5F2 publication path: the six leaves + manifest + deterministic PASS
    report persist at one shared run revision and the A5 CURRENT is published
    via a compare-and-set (``PointerKind.CURRENT``);
  * ``build_a5_semantic_identity``: the three domain preparations produce the
    exact backend-neutral A5 semantic identity (and fail closed on a
    profile / plan-hash disagreement);
  * the semantic-identity publication binding (a mismatched semantic identity
    fails closed before any artifact is written);
  * upstream A4 CURRENT stability: a stable A4 CURRENT publishes; an advanced
    A4 CURRENT fails closed (no A5 CURRENT, artifacts remain historical);
  * orphan revision skipping: a partial A5 revision-1 run is skipped;
  * corrupt A5 CURRENT fail-closed: a corrupt A5 leaf fails the typed loaders.

All fixtures are self-contained synthetic run trees (no provider calls). The A4
CURRENT is produced by the real A4 deterministic pipeline, so the A5F2
publication is exercised against a genuinely current-eligible A4 CURRENT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import (
    FilePointerStore,
    LineageRef,
    PointerKind,
    PointerNotFoundError,
    ValidationReport,
    persist_validation_report,
)
from short_drama.llm import LLMInvocationProvenance
from short_drama.llm.config import load_semantic_profile
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    A3InputIdentity,
    A5FinalizationResult,
    A5SemanticIdentity,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    CandidateExtraction,
    CandidatePayload,
    CanonicalEventSet,
    CanonicalFactSet,
    CanonicalRelationshipSet,
    CharacterCandidate,
    CHUNK_PLANNER_VERSION,
    ChunkManifest,
    ChunkPlanningProfile,
    ConsolidationPersistenceService,
    ConsolidationProfile,
    ConsolidationUpstreamUnstableError,
    EntityReconciliationProfile,
    EventCandidate,
    EventSemanticPreparation,
    EvidenceRef,
    FactCandidate,
    FactSemanticPreparation,
    LANGUAGE_DETECTOR_ID,
    LocationCandidate,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    PromptAssetIdentity,
    ReconciliationDecision,
    ReconciliationPersistenceService,
    ReconciliationSemanticResult,
    RelationshipCandidate,
    RelationshipSemanticPreparation,
    SourceChapter,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    StoryConflictSet,
    StoryExtractionProfile,
    StoryIntegrityError,
    TOKEN_COUNTER_ID,
    UnresolvedMentionCandidate,
    build_a4_semantic_identity,
    build_a5_semantic_identity,
    build_consolidation_planning,
    compute_llm_decision_id,
    finalize_reconciliation,
    load_consolidation_profile,
    load_entity_reconciliation_profile,
    llm_reason_code,
    pack_semantic_pairs_v1,
    plan_candidate_index_v1,
    plan_chunks,
    prepare_semantic_resolution,
    persist_candidate_extraction,
    persist_chunk_manifest,
    persist_source_chunk,
    persist_source_document,
    persist_consolidation_candidate_index,
)
from short_drama.story.persistence import (
    chunk_pointer_id,
    chunk_validation_artifact_id,
    source_pointer_id,
    source_validation_artifact_id,
)
from short_drama.story.reconciliation_planning import (
    _local_candidate_suffix,
    compute_source_order_key,
)
from short_drama.story.source import NormalizationInfo
from short_drama.story.consolidation_persistence import (
    a5_base_artifact_id,
    a5_pointer_id,
    a5_validation_artifact_id,
    canonical_fact_set_artifact_id,
    consolidation_candidate_index_artifact_id,
    consolidation_manifest_artifact_id,
)

PROJECT = "a5f2test"
DOCUMENT = "src_001"
CHUNK_PROFILE_ID = "story-analysis-v1"
EXTRACTION_PROFILE_ID = "story-extraction-v1"
RECON_PROFILE_ID = "entity-reconciliation-v2"
CONSOLIDATION_PROFILE_ID = "consolidation-v1"
RECON_PROFILE_PATH = REPO_ROOT / "profiles" / "entity_reconciliation_v2.yaml"
A4_LLM_PROFILE_PATH = REPO_ROOT / "profiles" / "entity_reconciliation_llm_v1.yaml"
CONSOLIDATION_PROFILE_PATH = REPO_ROOT / "profiles" / "consolidation_v1.yaml"

_PARAS = ("CH001_P0001", "CH001_P0002", "CH001_P0003")
_CHUNK_ID = "CH001_C001"
_HASH = "f" * 64
_HASH2 = "e" * 64


def _consolidation_profile() -> ConsolidationProfile:
    return load_consolidation_profile(CONSOLIDATION_PROFILE_PATH)


def _recon_profile() -> EntityReconciliationProfile:
    return load_entity_reconciliation_profile(RECON_PROFILE_PATH)


def _extraction_profile() -> StoryExtractionProfile:
    return StoryExtractionProfile(
        schema_version=1,
        profile_id=EXTRACTION_PROFILE_ID,
        working_language="zh-CN",
        prompt_id="a3.chunk-extraction",
        prompt_version=1,
        output_schema_id="a3-candidate-payload",
        output_schema_version=1,
        max_generation_rounds=2,
    )


def _chunk_profile() -> ChunkPlanningProfile:
    return ChunkPlanningProfile(
        schema_version=1,
        profile_id=CHUNK_PROFILE_ID,
        token_counter=TOKEN_COUNTER_ID,
        ownership_token_budget=100000,
        context_overlap_token_budget=0,
        context_token_budget=100000,
    )


def _provenance() -> LLMInvocationProvenance:
    return LLMInvocationProvenance(
        provider_family="qwen",
        model="qwen3-27b",
        semantic_profile_id="story-extraction-llm-v1",
        semantic_profile_hash=_HASH,
        prompt_id="a3.chunk-extraction",
        prompt_version=1,
        prompt_content_hash=_HASH,
        rendered_prompt_hash=_HASH,
        output_schema_id="a3-candidate-payload",
        output_schema_version=1,
        output_schema_hash=_HASH,
        request_hash=_HASH,
        provider_response_id=None,
        finish_reason="stop",
        usage={"total_tokens": 100},
    )


def _evidence(paragraph_id: str = _PARAS[0]) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role="primary", strength="explicit", excerpt="txt"
    )


def _char(candidate_id: str, name: str) -> CharacterCandidate:
    return CharacterCandidate(
        candidate_id=candidate_id,
        display_name_original=name,
        aliases_original=(),
        descriptors_zh=(name,),
        summary_zh=name,
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _loc(candidate_id: str, name: str) -> LocationCandidate:
    return LocationCandidate(
        candidate_id=candidate_id,
        display_name_original=name,
        aliases_original=(),
        descriptors_zh=(name,),
        summary_zh=name,
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _unres(candidate_id: str, kind: str) -> UnresolvedMentionCandidate:
    return UnresolvedMentionCandidate(
        candidate_id=candidate_id,
        mention_original="?",
        mention_kind=kind,
        reason_zh="unclear",
        possible_candidate_refs=(),
        evidence_strength="uncertain",
        evidence=(_evidence(),),
    )


def _fact(candidate_id: str, subject_refs=(), object_refs=()) -> FactCandidate:
    return FactCandidate(
        candidate_id=candidate_id,
        fact_type="world_fact",
        statement_zh="stmt",
        subject_refs=tuple(subject_refs),
        object_refs=tuple(object_refs),
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _event(
    candidate_id: str, participant_refs=(), location_refs=()
) -> EventCandidate:
    return EventCandidate(
        candidate_id=candidate_id,
        summary_zh="evt",
        participant_refs=tuple(participant_refs),
        location_refs=tuple(location_refs),
        temporal_mode="normal",
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _rel(candidate_id: str, source_ref: str, target_ref: str) -> RelationshipCandidate:
    return RelationshipCandidate(
        candidate_id=candidate_id,
        source_ref=source_ref,
        target_ref=target_ref,
        relationship_type_zh="meets",
        state_zh=None,
        direction="directed",
        evidence_strength="explicit",
        evidence=(_evidence(),),
    )


def _make_llm_provenance(prep, request_hash: str) -> LLMInvocationProvenance:
    return LLMInvocationProvenance(
        provider_family="qwen",
        model="qwen3-27b",
        semantic_profile_id=prep.semantic_profile_id,
        semantic_profile_hash=prep.semantic_profile_hash,
        prompt_id=prep.prompt_id,
        prompt_version=prep.prompt_version,
        prompt_content_hash=prep.prompt_content_hash,
        rendered_prompt_hash=_HASH,
        output_schema_id=prep.output_schema_id,
        output_schema_version=prep.output_schema_version,
        output_schema_hash=prep.output_schema_hash,
        request_hash=request_hash,
        provider_response_id="resp",
        finish_reason="stop",
        usage={},
    )


def _idx_entry(
    chunk_id: str, local_id: str, kind: str, ext_ref, name: str
) -> CandidateEntityIndexEntry:
    category = (
        "character"
        if kind == "character"
        else "location"
        if kind == "location"
        else "unresolved"
    )
    global_ref = f"{chunk_id}:{local_id}"
    order_key = compute_source_order_key(
        chunk_ordinal=1,
        paragraph_ordinal=1,
        category=category,
        candidate_suffix=_local_candidate_suffix(local_id),
        global_ref=global_ref,
    )
    return CandidateEntityIndexEntry(
        candidate_ref=global_ref,
        candidate_kind=kind,
        candidate_extraction_ref=ext_ref,
        source_order_key=order_key,
        display_name_original=name,
        aliases_original=(),
        descriptors_zh=(name,),
        evidence_refs=(_evidence(),),
        possible_candidate_refs=(),
    )


@dataclass
class BaseTree:
    """The shared A1+A2 state (source document + chunk manifest)."""

    store: FileArtifactStore
    pointers: FilePointerStore
    source_ref: object
    chunk: object
    chunk_ref: object
    manifest: ChunkManifest
    manifest_ref: object
    recon_profile: EntityReconciliationProfile
    consolidation_profile: ConsolidationProfile


def _build_base_tree(tmp_path: Path) -> BaseTree:
    """Persist the A1 source document and A2 chunk manifest (zero provider)."""
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)

    source_doc = SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt", "source/novel.txt", _HASH, 128, "zh-CN", "zh", LANGUAGE_DETECTOR_ID
        ),
        normalization=NormalizationInfo(
            "utf-8", "LF", "short_drama_source_ingestion_v1", "1"
        ),
        chapters=(
            SourceChapter(
                "CH001",
                None,
                "synthetic",
                tuple(SourceParagraph(pid, "text", None) for pid in _PARAS),
            ),
        ),
    )
    source_ref = persist_source_document(store, source_doc, revision=1)
    pointers.compare_and_set(
        pointer_id=source_pointer_id(PROJECT, DOCUMENT),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=source_ref,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", source_ref),), findings=()
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=source_ref.revision,
    )

    chunks, coverage = plan_chunks(source_doc, source_ref, _chunk_profile())
    assert len(chunks) == 1, "fixture assumes exactly one chunk"
    chunk = chunks[0]
    chunk_ref = persist_source_chunk(
        store, chunk, profile_id=CHUNK_PROFILE_ID, revision=1
    )
    manifest = ChunkManifest(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source_document_ref=source_ref,
        planner_version=CHUNK_PLANNER_VERSION,
        profile=_chunk_profile(),
        chunk_refs=(chunk_ref,),
        chunk_count=1,
        coverage=coverage,
        state="CHUNKING_COMPLETE",
    )
    manifest_ref = persist_chunk_manifest(store, manifest, revision=1)
    pointers.compare_and_set(
        pointer_id=chunk_pointer_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=manifest_ref,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(
                LineageRef("source_document", source_ref),
                LineageRef("chunk_manifest", manifest_ref),
            ),
            findings=(),
        ),
        artifact_id=chunk_validation_artifact_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID),
        revision=manifest_ref.revision,
    )

    return BaseTree(
        store=store,
        pointers=pointers,
        source_ref=source_ref,
        chunk=chunk,
        chunk_ref=chunk_ref,
        manifest=manifest,
        manifest_ref=manifest_ref,
        recon_profile=_recon_profile(),
        consolidation_profile=_consolidation_profile(),
    )


def _default_candidates(suffix: str = "001"):
    """A small, fully-connected candidate set (2 chars, 1 loc, 1 unres + fact/event/rel).

    ``suffix`` is a 3-digit run discriminator so that two candidate sets (e.g.
    ``001`` and ``002``) produce disjoint candidate ids (the candidate id
    namespace is ``cand_<kind>_[0-9]{3,}``).
    """
    char_1 = f"cand_char_{suffix}1"
    char_2 = f"cand_char_{suffix}2"
    loc_1 = f"cand_loc_{suffix}1"
    unres_1 = f"cand_unres_{suffix}1"
    fact_1 = f"cand_fact_{suffix}1"
    event_1 = f"cand_evt_{suffix}1"
    rel_1 = f"cand_rel_{suffix}1"
    chars = (_char(char_1, "Alice"), _char(char_2, "Bob"))
    locs = (_loc(loc_1, "Wonderland"),)
    unres = (_unres(unres_1, "unknown"),)
    facts = (_fact(fact_1, subject_refs=(char_1,), object_refs=(loc_1,)),)
    events = (
        _event(event_1, participant_refs=(char_1,), location_refs=(loc_1,)),
    )
    rels = (_rel(rel_1, char_1, char_2),)
    return chars, locs, unres, facts, events, rels


def _publish_a4(tree: BaseTree, *, suffix: str = "001", revision: int = 1):
    """Persist the A3 candidate extraction and publish a valid A4 CURRENT.

    Uses the deterministic A4 pipeline (zero provider calls); uncertain pairs
    get a deterministic "llm" different_entity decision bound to the exact
    block request hash. ``revision`` is the candidate-extraction run revision
    (a second candidate set for the same chunk/profile uses revision 2 so the
    A3 input identity differs). Returns ``(ext_ref, a3_input, index)``.
    """
    store = tree.store
    pointers = tree.pointers
    recon_profile = tree.recon_profile
    chars, locs, unres, facts, events, rels = _default_candidates(suffix)

    ext_profile = _extraction_profile()
    extraction = CandidateExtraction(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile_id=CHUNK_PROFILE_ID,
        chunk_id=_CHUNK_ID,
        source_document_ref=tree.source_ref,
        source_chunk_ref=tree.chunk_ref,
        extraction_profile_id=ext_profile.profile_id,
        extraction_profile_hash=ext_profile.profile_hash,
        generation_provenance=_provenance(),
        candidates=CandidatePayload(
            characters=tuple(chars),
            locations=tuple(locs),
            facts=tuple(facts),
            events=tuple(events),
            relationships=tuple(rels),
            unresolved_mentions=tuple(unres),
        ),
    )
    ext_ref = persist_candidate_extraction(store, extraction, revision=revision)

    a3_input = A3InputIdentity(
        source_document_ref=tree.source_ref,
        chunk_manifest_ref=tree.manifest_ref,
        candidate_extraction_refs=(ext_ref,),
        extraction_profile_id=ext_profile.profile_id,
        extraction_profile_hash=ext_profile.profile_hash,
    )
    index_entries = tuple(
        _idx_entry(_CHUNK_ID, c.candidate_id, "character", ext_ref, c.display_name_original)
        for c in chars
    ) + tuple(
        _idx_entry(_CHUNK_ID, c.candidate_id, "location", ext_ref, c.display_name_original)
        for c in locs
    ) + tuple(
        _idx_entry(_CHUNK_ID, c.candidate_id, f"unresolved_{c.mention_kind}", ext_ref, c.mention_original)
        for c in unres
    )
    index = CandidateEntityIndex(schema_version=1, entries=index_entries)

    # Deterministic A4 pipeline: plan -> finalize -> identity -> publish.
    planning = plan_candidate_index_v1(index)
    sem_profile = load_semantic_profile(A4_LLM_PROFILE_PATH)
    prep = prepare_semantic_resolution(planning, recon_profile, sem_profile)
    request_hashes = prep.semantic_request_hashes
    pair_to_hash: dict = {}
    for block_ordinal, block in enumerate(pack_semantic_pairs_v1(planning)):
        for plan in block:
            pair_to_hash[(plan.left_candidate_ref, plan.right_candidate_ref)] = (
                request_hashes[block_ordinal]
            )
    llm_decisions = [
        ReconciliationDecision(
            decision_id=compute_llm_decision_id(
                left_ref=plan.left_candidate_ref,
                right_ref=plan.right_candidate_ref,
                decision="different_entity",
                method="llm",
                reason_code=llm_reason_code("different_entity"),
                reason_zh="different",
                evidence_refs=(),
                prompt_id=prep.prompt_id,
                prompt_version=prep.prompt_version,
                request_hash=pair_to_hash[
                    (plan.left_candidate_ref, plan.right_candidate_ref)
                ],
            ),
            left_candidate_ref=plan.left_candidate_ref,
            right_candidate_ref=plan.right_candidate_ref,
            decision="different_entity",
            method="llm",
            reason_code=llm_reason_code("different_entity"),
            reason_zh="different",
            evidence_refs=(),
            prompt_id=prep.prompt_id,
            prompt_version=prep.prompt_version,
            generation_provenance=_make_llm_provenance(
                prep, pair_to_hash[(plan.left_candidate_ref, plan.right_candidate_ref)]
            ),
        )
        for plan in planning.pair_plans
        if plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
    ]
    all_decisions = planning.decisions + tuple(llm_decisions)
    semantic_result = ReconciliationSemanticResult(
        planning_result=planning,
        blocks=prep.blocks,
        semantic_decisions=tuple(llm_decisions),
        all_decisions=all_decisions,
        semantic_request_hashes=request_hashes,
        block_results=(),
    )
    finalization = finalize_reconciliation(semantic_result)
    identity = build_a4_semantic_identity(recon_profile, prep, planning)
    ReconciliationPersistenceService(store, pointers).publish_validated(
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile=recon_profile,
        semantic_identity=identity,
        a3_input=a3_input,
        finalization_result=finalization,
    )
    return ext_ref, a3_input, index


def _build_tree_with_a4(tmp_path: Path, *, suffix: str = "001"):
    """Build the A1+A2 base tree and publish a valid A4 CURRENT (one candidate set)."""
    tree = _build_base_tree(tmp_path)
    ext_ref, a3_input, index = _publish_a4(tree, suffix=suffix)
    return tree, ext_ref, a3_input, index


# ---------------------------------------------------------------------------
# A5F2 input construction
# ---------------------------------------------------------------------------


def _planning(tree: BaseTree):
    return build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=tree.consolidation_profile,
    )


def _make_preparations(planning, consolidation_profile):
    """Build the three domain semantic preparations (zero semantic blocks).

    These are structurally valid preparations with no semantic pairs (a small
    candidate set produces no needs-semantic pairs). The prompt / output-schema
    asset identities use the tracked A5C/A5D ids/versions and a fixed content
    hash; the semantic profile is the tracked ``consolidation-llm-v1``.
    """
    from short_drama.story import (
        A5C_FACT_OUTPUT_SCHEMA_ID,
        A5C_FACT_OUTPUT_SCHEMA_VERSION,
        A5C_FACT_PROMPT_ID,
        A5C_FACT_PROMPT_VERSION,
        A5D_EVENT_OUTPUT_SCHEMA_ID,
        A5D_EVENT_OUTPUT_SCHEMA_VERSION,
        A5D_EVENT_PROMPT_ID,
        A5D_EVENT_PROMPT_VERSION,
        A5D_RELATIONSHIP_OUTPUT_SCHEMA_ID,
        A5D_RELATIONSHIP_OUTPUT_SCHEMA_VERSION,
        A5D_RELATIONSHIP_PROMPT_ID,
        A5D_RELATIONSHIP_PROMPT_VERSION,
        EVENT_SEMANTIC_PACKING_V1,
        FACT_SEMANTIC_PACKING_V1,
        RELATIONSHIP_SEMANTIC_PACKING_V1,
        load_fact_semantic_profile,
        load_relationship_semantic_profile,
        load_event_semantic_profile,
    )

    fact_sem = load_fact_semantic_profile()
    event_sem = load_event_semantic_profile()
    rel_sem = load_relationship_semantic_profile()
    common = dict(
        planning_result=planning,
        consolidation_profile=consolidation_profile,
        working_language="zh-CN",
        blocks=(),
        structured_requests=(),
        semantic_request_hashes=(),
        semantic_pair_count=0,
        auto_same_pair_count=0,
    )
    fact_prep = FactSemanticPreparation(
        semantic_profile=fact_sem,
        prompt_id=A5C_FACT_PROMPT_ID,
        prompt_version=A5C_FACT_PROMPT_VERSION,
        prompt_content_hash=_HASH,
        output_schema_id=A5C_FACT_OUTPUT_SCHEMA_ID,
        output_schema_version=A5C_FACT_OUTPUT_SCHEMA_VERSION,
        output_schema_hash=_HASH,
        packing_policy=FACT_SEMANTIC_PACKING_V1,
        total_fact_pair_count=0,
        **common,
    )
    event_prep = EventSemanticPreparation(
        semantic_profile=event_sem,
        prompt_id=A5D_EVENT_PROMPT_ID,
        prompt_version=A5D_EVENT_PROMPT_VERSION,
        prompt_content_hash=_HASH,
        output_schema_id=A5D_EVENT_OUTPUT_SCHEMA_ID,
        output_schema_version=A5D_EVENT_OUTPUT_SCHEMA_VERSION,
        output_schema_hash=_HASH,
        packing_policy=EVENT_SEMANTIC_PACKING_V1,
        total_event_pair_count=0,
        **common,
    )
    rel_prep = RelationshipSemanticPreparation(
        semantic_profile=rel_sem,
        prompt_id=A5D_RELATIONSHIP_PROMPT_ID,
        prompt_version=A5D_RELATIONSHIP_PROMPT_VERSION,
        prompt_content_hash=_HASH,
        output_schema_id=A5D_RELATIONSHIP_OUTPUT_SCHEMA_ID,
        output_schema_version=A5D_RELATIONSHIP_OUTPUT_SCHEMA_VERSION,
        output_schema_hash=_HASH,
        packing_policy=RELATIONSHIP_SEMANTIC_PACKING_V1,
        total_relationship_pair_count=0,
        **common,
    )
    return fact_prep, event_prep, rel_prep


def _empty_finalization(planning) -> A5FinalizationResult:
    """An A5 finalization result with empty canonical sets (a small candidate
    set produces no canonical facts / events / relationships / conflicts)."""
    return A5FinalizationResult(
        planning_result=planning,
        canonical_fact_set=CanonicalFactSet(schema_version=1, facts=(), state_transitions=()),
        canonical_event_set=CanonicalEventSet(schema_version=1, events=()),
        canonical_relationship_set=CanonicalRelationshipSet(schema_version=1, relationships=()),
        story_conflict_set=StoryConflictSet(schema_version=1, conflicts=()),
    )


def _publish_a5(
    tree: BaseTree,
    planning,
    *,
    semantic_identity=None,
    decision_set=None,
    service=None,
):
    """Drive the A5F2 ``publish_validated`` path with the constructed inputs."""
    service = service or ConsolidationPersistenceService(tree.store, tree.pointers)
    if semantic_identity is None:
        fact_prep, event_prep, rel_prep = _make_preparations(
            planning, tree.consolidation_profile
        )
        semantic_identity = build_a5_semantic_identity(fact_prep, event_prep, rel_prep)
    if decision_set is None:
        decision_set = planning.deterministic_decision_set
    return service.publish_validated(
        project_id=PROJECT,
        document_id=DOCUMENT,
        consolidation_profile=tree.consolidation_profile,
        semantic_identity=semantic_identity,
        finalization_result=_empty_finalization(planning),
        decision_set=decision_set,
    )


def _current_a5_target(tree: BaseTree):
    try:
        pointer = tree.pointers.resolve_current(a5_pointer_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID))
        return pointer.target_ref
    except PointerNotFoundError:
        return None


def _store_path(store: FileArtifactStore, artifact_type: str, artifact_id: str, revision: int) -> Path:
    return store.root / artifact_type / artifact_id / f"r{revision:08d}.json"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_artifact_identity_matches_contract():
    base = a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    assert base == f"{PROJECT}.{DOCUMENT}.a5.{CONSOLIDATION_PROFILE_ID}"
    assert (
        consolidation_candidate_index_artifact_id(base)
        == f"{base}.candidate-index"
    )
    assert consolidation_manifest_artifact_id(base) == f"{base}.manifest"
    assert a5_validation_artifact_id(base) == f"{base}.manifest.a5-validation"
    assert (
        a5_pointer_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
        == f"{PROJECT}.a5.{DOCUMENT}.{CONSOLIDATION_PROFILE_ID}"
    )


def test_build_a5_semantic_identity_matches_preparations(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    fact_prep, event_prep, rel_prep = _make_preparations(
        planning, tree.consolidation_profile
    )
    identity = build_a5_semantic_identity(fact_prep, event_prep, rel_prep)

    assert isinstance(identity, A5SemanticIdentity)
    assert identity.consolidation_profile_id == tree.consolidation_profile.profile_id
    assert (
        identity.consolidation_profile_hash
        == tree.consolidation_profile.content_hash()
    )
    assert identity.plan_hash == planning.plan_hash
    # Domain-ordered prompt / output-schema asset identities.
    assert identity.prompt_identities == (
        PromptAssetIdentity(
            prompt_id=fact_prep.prompt_id,
            prompt_version=fact_prep.prompt_version,
            prompt_content_hash=fact_prep.prompt_content_hash,
        ),
        PromptAssetIdentity(
            prompt_id=event_prep.prompt_id,
            prompt_version=event_prep.prompt_version,
            prompt_content_hash=event_prep.prompt_content_hash,
        ),
        PromptAssetIdentity(
            prompt_id=rel_prep.prompt_id,
            prompt_version=rel_prep.prompt_version,
            prompt_content_hash=rel_prep.prompt_content_hash,
        ),
    )
    assert len(identity.output_schema_identities) == 3
    # Zero semantic pairs -> no semantic request hashes.
    assert identity.semantic_request_hashes == ()


def test_build_a5_semantic_identity_disagreement_fails_closed(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    fact_prep, event_prep, rel_prep = _make_preparations(
        planning, tree.consolidation_profile
    )
    # A different consolidation profile in one domain -> fail closed.
    other_profile = ConsolidationProfile(
        schema_version=1,
        profile_id="other-consolidation-profile",
        working_language="zh-CN",
        blocking_policy_id="consolidation-blocking-v1",
        max_generation_rounds=1,
        fact=tree.consolidation_profile.fact,
        event=tree.consolidation_profile.event,
        relationship=tree.consolidation_profile.relationship,
    )
    from dataclasses import replace

    other_event_prep = replace(event_prep, consolidation_profile=other_profile)
    with pytest.raises(StoryIntegrityError, match="disagree on the consolidation profile id"):
        build_a5_semantic_identity(fact_prep, other_event_prep, rel_prep)


# ---------------------------------------------------------------------------
# Validated publication
# ---------------------------------------------------------------------------


def test_first_valid_publish(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    pub = _publish_a5(tree, planning)

    assert pub.reused is False
    # All eight A5 slots share one run revision.
    refs = (
        pub.consolidation_candidate_index_ref,
        pub.consolidation_decision_set_ref,
        pub.canonical_fact_set_ref,
        pub.canonical_event_set_ref,
        pub.canonical_relationship_set_ref,
        pub.story_conflict_set_ref,
        pub.consolidation_manifest_ref,
        pub.validation_report_ref,
    )
    assert all(ref.revision == 1 for ref in refs)
    base = a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    assert pub.consolidation_manifest_ref.artifact_id == consolidation_manifest_artifact_id(base)
    # The PASS validation report is persisted and PASS.
    from short_drama.foundation import load_validation_report, ValidationResult

    report = load_validation_report(tree.store, pub.validation_report_ref)
    assert report.summary.result is ValidationResult.PASS
    # The A5 CURRENT pointer is published at CURRENT.
    pointer = tree.pointers.resolve_current(
        a5_pointer_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    )
    assert pointer.pointer_kind is PointerKind.CURRENT
    assert pointer.target_ref == pub.consolidation_manifest_ref
    assert pub.current_pointer_ref == pub.consolidation_manifest_ref


def test_publication_pins_exact_a4_and_a3(tmp_path):
    tree, _ext_ref, a3_input, _ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    pub = _publish_a5(tree, planning)

    from short_drama.story import load_consolidation_manifest

    manifest = load_consolidation_manifest(
        tree.store,
        pub.consolidation_manifest_ref,
        expected_artifact_id=consolidation_manifest_artifact_id(
            a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
        ),
    )
    # The manifest pins the exact A4 EntityMap and A3 input identity.
    assert manifest.entity_map_ref == planning.snapshot.entity_map_ref
    assert manifest.upstream_identity.a3_input == a3_input
    assert manifest.project_id == PROJECT
    assert manifest.document_id == DOCUMENT
    # The six leaves share the manifest run revision and are the published refs.
    assert manifest.consolidation_candidate_index_ref == pub.consolidation_candidate_index_ref
    assert manifest.canonical_fact_set_ref == pub.canonical_fact_set_ref
    assert manifest.canonical_event_set_ref == pub.canonical_event_set_ref
    assert manifest.canonical_relationship_set_ref == pub.canonical_relationship_set_ref
    assert manifest.story_conflict_set_ref == pub.story_conflict_set_ref


def test_republish_advances_revision(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    first = _publish_a5(tree, planning)
    assert first.consolidation_manifest_ref.revision == 1
    second = _publish_a5(tree, planning)
    assert second.consolidation_manifest_ref.revision == 2
    assert second.reused is False
    # The CURRENT pointer advances to the new manifest.
    pointer = tree.pointers.resolve_current(
        a5_pointer_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    )
    assert pointer.target_ref == second.consolidation_manifest_ref


# ---------------------------------------------------------------------------
# Upstream A4 stability
# ---------------------------------------------------------------------------


def test_upstream_a4_advanced_fails_closed(tmp_path):
    # Build the A5 planning against the first A4 CURRENT (candidate set "001").
    tree, *_ = _build_tree_with_a4(tmp_path, suffix="001")
    planning = _planning(tree)
    # Advance the A4 CURRENT with a DIFFERENT candidate set (different A3 input,
    # candidate extraction revision 2 so the A3 input identity differs).
    _publish_a4(tree, suffix="002", revision=2)
    with pytest.raises(ConsolidationUpstreamUnstableError):
        _publish_a5(tree, planning)
    # No A5 CURRENT was published. The A4 stability re-check runs AFTER the run
    # is persisted and BEFORE the CAS, so the orphaned revision-1 run remains
    # historical (never pointed to).
    assert _current_a5_target(tree) is None
    base = a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    tree.store.get(
        "consolidation_manifest", consolidation_manifest_artifact_id(base), 1
    )


def test_upstream_a4_stable_publishes(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path, suffix="001")
    planning = _planning(tree)
    pub = _publish_a5(tree, planning)
    assert pub.reused is False
    assert _current_a5_target(tree) == pub.consolidation_manifest_ref


# ---------------------------------------------------------------------------
# Orphan skipping + corrupt fail-closed
# ---------------------------------------------------------------------------


def test_orphan_revision_skip(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    base = a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    # Simulate a failed publication that left a partial revision-1 candidate index
    # (an orphan never pointed to).
    persist_consolidation_candidate_index(
        tree.store,
        planning.index,
        artifact_id=consolidation_candidate_index_artifact_id(base),
        revision=1,
    )
    pub = _publish_a5(tree, planning)
    # The orphaned revision-1 slot is skipped; the run lands on revision 2.
    assert pub.consolidation_manifest_ref.revision == 2
    assert _current_a5_target(tree) == pub.consolidation_manifest_ref


def test_corrupt_a5_leaf_fails_closed(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    pub = _publish_a5(tree, planning)
    # Corrupt a pinned output artifact (the canonical fact set). The typed loader
    # must fail closed.
    path = _store_path(
        tree.store,
        "canonical_fact_set",
        canonical_fact_set_artifact_id(a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)),
        pub.canonical_fact_set_ref.revision,
    )
    path.write_bytes(b"this is not a valid artifact envelope")
    from short_drama.story import load_canonical_fact_set

    with pytest.raises(StoryIntegrityError):
        load_canonical_fact_set(
            tree.store,
            pub.canonical_fact_set_ref,
            expected_artifact_id=canonical_fact_set_artifact_id(
                a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
            ),
        )


# ---------------------------------------------------------------------------
# Semantic-identity binding
# ---------------------------------------------------------------------------


def test_semantic_identity_binding_mismatch_fails_closed(tmp_path):
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    fact_prep, event_prep, rel_prep = _make_preparations(
        planning, tree.consolidation_profile
    )
    identity = build_a5_semantic_identity(fact_prep, event_prep, rel_prep)
    # A semantic identity whose plan hash does not match the planning result ->
    # the binding fails closed before any artifact is written.
    from dataclasses import replace

    bad_identity = replace(identity, plan_hash=_HASH2)
    base = a5_base_artifact_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID)
    with pytest.raises(StoryIntegrityError, match="plan_hash"):
        _publish_a5(tree, planning, semantic_identity=bad_identity)
    # No A5 artifact was written and no A5 CURRENT was published.
    assert _current_a5_target(tree) is None
    with pytest.raises(Exception):
        tree.store.get("consolidation_manifest", consolidation_manifest_artifact_id(base), 1)
